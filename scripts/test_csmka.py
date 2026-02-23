import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.knowledge_base.build_knowledge_base import KnowledgeBaseBuilder
from src.alignment.knowledge_alignment import KnowledgeAlignment

def evaluate_model():
    """评估训练好的模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 模型路径
    base_model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
    adapter_path = os.path.join(base_dir, "output", "lora_model", "checkpoint-63")
    offload_dir = os.path.join(base_dir, "output", "offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    print(f"Base model path: {base_model_path}")
    print(f"Adapter path: {adapter_path}")
    print(f"Base model exists: {os.path.exists(base_model_path)}")
    print(f"Adapter exists: {os.path.exists(adapter_path)}")
    
    # 加载知识库
    print("\nLoading knowledge base...")
    output_dir = os.path.join(base_dir, "output")
    index_path = os.path.join(output_dir, "knowledge_base.index")
    metadata_path = os.path.join(output_dir, "knowledge_base_metadata.json")
    
    try:
        knowledge_base = KnowledgeBaseBuilder()
        knowledge_base.load(index_path, metadata_path)
        print(f"Knowledge base loaded successfully! Contains {len(knowledge_base.metadata)} records")
        alignment = KnowledgeAlignment(knowledge_base)
    except Exception as e:
        print(f"Failed to load knowledge base: {e}")
        knowledge_base = None
        alignment = None
    
    # 加载tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto",
        offload_folder=offload_dir,
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        dtype=torch.float16
    )
    
    # 设置为评估模式
    model.eval()
    
    # 测试样本
    test_samples = [
        {"instruction": "什么是表闭水停证？", "input": ""},
        {"instruction": "什么是肺炎？", "input": ""},
        {"instruction": "请解释中医的阴阳理论", "input": ""},
        {"instruction": "如何预防感冒？", "input": ""},
        {"instruction": "什么是针灸？", "input": ""}
    ]
    
    print("\n开始评估模型...")
    print("=" * 80)
    
    for i, sample in enumerate(test_samples):
        instruction = sample["instruction"]
        input_text = sample["input"]
        
        # 从知识库检索相关信息
        evidence = []
        if alignment:
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
        
        print(f"\n测试样本 {i+1}:")
        print(f"指令: {instruction}")
        if input_text:
            print(f"输入: {input_text}")
        
        # 生成回复
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
            response = response.split("### Response:\n")[-1].strip()
        
        print(f"回复: {response}")
        print("-" * 80)
    
    print("\n评估完成！")

if __name__ == "__main__":
    evaluate_model()
