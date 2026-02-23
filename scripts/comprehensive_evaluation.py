import os
import sys
import torch
import json
import numpy as np
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.knowledge_base.build_knowledge_base import KnowledgeBaseBuilder
from src.alignment.knowledge_alignment import KnowledgeAlignment

def load_model(model_type):
    """加载不同类型的模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offload_dir = os.path.join(base_dir, "output", "offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    if model_type == "baseline":
        # 基线模型 - 原始Qwen3-1.7B
        model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
        print(f"Loading Base model: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,
            trust_remote_code=True
        )
        
    elif model_type == "csmka":
        # CSMKA模型 - 结合了RAG的模型
        base_model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
        adapter_path = os.path.join(base_dir, "output", "lora_model", "checkpoint-63")
        print(f"加载CSMKA模型: {adapter_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            dtype=torch.float16
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    model.eval()
    return model, tokenizer

def calculate_perplexity(model, tokenizer, texts):
    """计算困惑度"""
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"计算困惑度时出错: {e}")
        return float('inf')

def generate_response(model, tokenizer, instruction, input_text="", alignment=None, model_type=""):
    """生成模型回复"""
    # 从知识库检索相关信息
    evidence = []
    if model_type == "csmka" and alignment:
        try:
            evidence = alignment.query(instruction, top_k=5)
            print(f"从知识库检索到 {len(evidence)} 条相关信息")
            # 打印检索到的证据内容
            for i, item in enumerate(evidence):
                content = item.get('content', '')
                score = item.get('score', 0)
                print(f"证据 {i+1} (分数: {score:.4f}): {content[:100]}...")
        except Exception as e:
            print(f"检索知识库失败: {e}")
    
    # 构建提示词
    # 强制要求模型使用中文回复
    prompt = f"### Instruction:\n请你必须使用中文回答以下问题，不要使用任何英文：{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    
    # 添加知识库证据
    if evidence:
        prompt += "### Evidence:\n"
        for i, item in enumerate(evidence[:3]):  # 只使用前3条证据
            content = item.get('content', '')
            if content:
                prompt += f"{i+1}. {content}\n"
    
    prompt += "### Response:\n"
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取回复部分
        if "### Response:\n" in response:
            response = response.split("### Response:\n")[-1].strip()
        # 移除可能的重复内容
        if "\n\n" in response:
            parts = response.split("\n\n")
            response = parts[0]
        # 移除可能的英文前缀
        if response.startswith("The "):
            response = response.split("\n")[0]
        # 确保返回的是字符串
        if not isinstance(response, str):
            response = str(response)
    
    # 计算实际词数
    # 使用更可靠的方式分割文本
    import re
    words = re.findall(r'\b\w+\b', response)
    actual_length = len(words)
    print(f"生成的回复长度: {actual_length} 词")
    # 打印前几个词作为调试
    if words:
        print(f"回复前5个词: {words[:5]}")
    return response

def evaluate_models():
    """评估模型的性能"""
    # 测试样本
    test_samples = [
        {"instruction": "什么是表闭水停证？", "input": ""},
        {"instruction": "什么是肺炎？", "input": ""},
        {"instruction": "请解释中医的阴阳理论", "input": ""},
        {"instruction": "如何预防感冒？", "input": ""},
        {"instruction": "什么是针灸？", "input": ""}
    ]
    
    # 加载知识库
    print("\n加载知识库...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    index_path = os.path.join(output_dir, "knowledge_base.index")
    metadata_path = os.path.join(output_dir, "knowledge_base_metadata.json")
    
    try:
        knowledge_base = KnowledgeBaseBuilder()
        knowledge_base.load(index_path, metadata_path)
        print(f"知识库加载成功！包含 {len(knowledge_base.metadata)} 条记录")
        alignment = KnowledgeAlignment(knowledge_base)
    except Exception as e:
        print(f"加载知识库失败: {e}")
        knowledge_base = None
        alignment = None
    
    # 加载模型
    models = {}
    tokenizers = {}
    
    model_types = ["baseline", "csmka"]
    for model_type in model_types:
        try:
            model, tokenizer = load_model(model_type)
            models[model_type] = model
            tokenizers[model_type] = tokenizer
            print(f"成功加载 {model_type} 模型")
        except Exception as e:
            print(f"加载 {model_type} 模型失败: {e}")
    
    # 计算评估指标
    metrics = defaultdict(dict)
    
    # 1. 困惑度评估
    print("\n计算困惑度...")
    eval_texts = [sample["instruction"] for sample in test_samples]
    for model_type, model in models.items():
        perplexity = calculate_perplexity(model, tokenizers[model_type], eval_texts)
        metrics[model_type]["perplexity"] = perplexity
        print(f"{model_type} 模型困惑度: {perplexity:.4f}")
    
    # 生成响应
    print("\n生成模型响应...")
    generated_responses = defaultdict(list)
    
    # 准确答案
    correct_answers = {
        "什么是表闭水停证？": "表闭水停证是中医中的一个术语，因外邪束表，腠理闭塞，水道不利，水湿内停所致。临床以突发头面、肢体浮肿，无汗，小便不利，头痛，关节酸楚，舌苔白，脉浮，或伴见发热，恶风，畏寒，咽喉疼痛等为特征的证候。",
        "什么是肺炎？": "肺炎是肺部的炎症，通常由细菌、病毒或真菌引起。常见症状包括咳嗽、发热、呼吸困难、胸痛等。严重的肺炎可能需要住院治疗，使用抗生素或抗病毒药物。",
        "请解释中医的阴阳理论": "中医的阴阳理论是一种古代哲学思想，认为自然界和人体都由对立统一的两个方面构成。阴 阳概念源于古人观察自然现象时发现的两种基本对立而又相互依存的现象：阳代表积极、温暖、主 动的属性，",
    }
    
    for model_type, model in models.items():
        print(f"\n{model_type} 模型生成结果:")
        print("-" * 80)
        
        for i, sample in enumerate(test_samples[:3]):  # 只评估前3个样本
            instruction = sample["instruction"]
            input_text = sample["input"]
            
            response = generate_response(model, tokenizers[model_type], instruction, input_text, alignment, model_type)
            generated_responses[model_type].append(response)
            
            print(f"测试样本 {i+1}: {instruction}")
            print(f"回复: {response[:100]}..." if len(response) > 100 else f"回复: {response}")
            print("-" * 40)
    
    # 计算响应长度
    print("\n计算响应长度...")
    import re
    for model_type, responses in generated_responses.items():
        # 计算每个响应的词数
        lengths = []
        for resp in responses:
            # 确保resp是字符串
            if isinstance(resp, str):
                # 使用正则表达式分割文本
                words = re.findall(r'\b\w+\b', resp)
                lengths.append(len(words))
            else:
                lengths.append(0)
        
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        metrics[model_type]["average_response_length"] = avg_length
        print(f"{model_type} 模型平均响应长度: {avg_length:.2f} 词")
        # 打印每个响应的实际长度
        for i, length in enumerate(lengths):
            print(f"  测试样本 {i+1} 响应长度: {length} 词")
    
    # 计算词汇多样性
    print("\n计算词汇多样性...")
    for model_type, responses in generated_responses.items():
        all_words = []
        for resp in responses:
            all_words.extend(resp.split())
        unique_words = set(all_words)
        base_diversity = len(unique_words) / len(all_words) if all_words else 0
        # 依据平均响应长度（权重0.4）+现有算法
        avg_length = metrics[model_type]["average_response_length"]
        # 长度因子：理想长度在50-150词之间
        length_factor = 1.0
        if avg_length < 50:
            length_factor = avg_length / 50
        elif avg_length > 150:
            length_factor = 150 / avg_length
        lexical_diversity = base_diversity * 0.6 + length_factor * 0.4
        metrics[model_type]["lexical_diversity"] = lexical_diversity
        print(f"{model_type} 模型词汇多样性: {lexical_diversity:.4f}")
    
    # 计算指令遵循度
    print("\n评估指令遵循度...")
    for model_type, responses in generated_responses.items():
        relevance_scores = []
        for i, (resp, sample) in enumerate(zip(responses, test_samples[:3])):
            instruction = sample["instruction"]
            correct_answer = correct_answers.get(instruction, "")
            
            # 计算文本相似度
            # 方法1：计算相同词汇的比例
            resp_words = set(resp.split())
            answer_words = set(correct_answer.split())
            common_words = resp_words.intersection(answer_words)
            similarity = len(common_words) / len(answer_words) if answer_words else 0
            
            # 方法2：计算文本长度比例
            length_ratio = min(len(resp), len(correct_answer)) / max(len(resp), len(correct_answer)) if max(len(resp), len(correct_answer)) > 0 else 0
            
            # 综合相似度
            content_score = (similarity + length_ratio) / 2
            
            # 综合困惑度（权重0.3）
            perplexity = metrics[model_type]["perplexity"]
            # 困惑度归一化（取对数后归一化）
            perplexity_norm = 1 / (1 + np.log(perplexity) / 20)  # 归一化到0-1
            
            instruction_following = content_score * 0.7 + perplexity_norm * 0.6
            relevance_scores.append(instruction_following)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        metrics[model_type]["instruction_following"] = avg_relevance
        print(f"{model_type} 模型指令遵循度: {avg_relevance:.4f}")
    
    # 计算生成质量评估
    print("\n计算生成质量评估...")
    for model_type, responses in generated_responses.items():
        perplexity = metrics[model_type]["perplexity"]
        perplexity_norm = 1 / (1 + np.log(perplexity) / 20)  # 归一化到0-1
        
        avg_length = metrics[model_type]["average_response_length"]
        # 长度归一化
        length_norm = min(avg_length / 100, 1.0) if avg_length > 0 else 0
        
        lexical_diversity = metrics[model_type]["lexical_diversity"]
        
        # 综合计算生成质量
        generation_quality = perplexity_norm * 0.3 + length_norm * 0.3 + lexical_diversity * 0.4
        metrics[model_type]["generation_quality"] = generation_quality
        print(f"{model_type} 模型生成质量: {generation_quality:.4f}")
    
    # 保存评估结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    evaluation_result_path = os.path.join(output_dir, "model_evaluation_results.json")
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存到: {evaluation_result_path}")
    
    # 打印最终评估结果
    print("\n" + "=" * 120)
    print("最终评估结果")
    print("=" * 120)
    print("模型类型\t\t困惑度\t\t平均响应长度\t词汇多样性\t指令遵循度\t生成质量")
    print("-" * 120)
    
    for model_type in model_types:
        if model_type in metrics:
            metric = metrics[model_type]
            print(f"{model_type}\t\t{metric.get('perplexity', 'N/A'):.4f}\t{metric.get('average_response_length', 'N/A'):.2f}\t\t{metric.get('lexical_diversity', 'N/A'):.4f}\t\t{metric.get('instruction_following', 'N/A'):.4f}\t{metric.get('generation_quality', 'N/A'):.4f}")
    
    print("=" * 100)
    
    # 释放内存
    for model in models.values():
        del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate_models()
