import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.knowledge_base.build_knowledge_base import KnowledgeBaseBuilder
from src.alignment.knowledge_alignment import KnowledgeAlignment

def load_model(model_type):
    """Load different types of models"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offload_dir = os.path.join(base_dir, "output", "offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    if model_type == "baseline":
        # Baseline model - original Qwen3-1.7B
        model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
        print(f"Loading baseline model: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,
            trust_remote_code=True
        )
        
    elif model_type == "csmka":
        # CSMKA model - fine-tuned model with knowledge base integration
        base_model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
        adapter_path = os.path.join(base_dir, "output", "lora_model", "checkpoint-63")
        print(f"Loading CSMKA model: {adapter_path}")
        
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
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model, tokenizer

def calculate_perplexity(model, tokenizer, texts):
    """Calculate perplexity"""
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')

def generate_response(model, tokenizer, instruction, input_text="", alignment=None, model_type=""):
    """Generate model response"""
    # Retrieve relevant information from knowledge base
    evidence = []
    if model_type == "csmka" and alignment:
        try:
            evidence = alignment.query(instruction, top_k=5)
            print(f"Retrieved {len(evidence)} relevant information from knowledge base")
            # Print retrieved evidence content
            for i, item in enumerate(evidence):
                content = item.get('content', '')
                score = item.get('score', 0)
                print(f"Evidence {i+1} (score: {score:.4f}): {content[:100]}...")
        except Exception as e:
            print(f"Failed to retrieve from knowledge base: {e}")
    
    # Build prompt
    # 强制要求模型使用中文回复
    prompt = f"### Instruction:\n请你必须使用中文回答以下问题，不要使用任何英文：{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    
    # Add knowledge base evidence
    if evidence:
        prompt += "### Evidence:\n"
        for i, item in enumerate(evidence[:3]):  # Only use first 3 pieces of evidence
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
        response = response.split("### Response:\n")[-1].strip()
    
    return response

def calculate_metrics(models, tokenizers, test_samples, alignment=None):
    """Calculate evaluation metrics for all models"""
    metrics = defaultdict(dict)
    generated_responses = defaultdict(list)
    
    # 准确答案
    correct_answers = {
        "什么是表闭水停证？": "表闭水停证是中医中的一个术语，因外邪束表，腠理闭塞，水道不利，水湿内停所致。临床以突发头面、肢体浮肿，无汗，小便不利，头痛，关节酸楚，舌苔白，脉浮，或伴见发热，恶风，畏寒，咽喉疼痛等为特征的证候。",
        "什么是肺炎？": "肺炎是肺部的炎症，通常由细菌、病毒或真菌引起。常见症状包括咳嗽、发热、呼吸困难、胸痛等。严重的肺炎可能需要住院治疗，使用抗生素或抗病毒药物。",
        "请解释中医的阴阳理论": "中医的阴阳理论是一种古老的哲学思想体系，它认为世界万物都由阴阳两种对立统一的基本属性构成。阴阳相互依存、相互转化、相互作用，构成了自然界和人体的动态平衡。",
    }
    
    # Calculate perplexity
    print("\nCalculating perplexity...")
    eval_texts = [sample["instruction"] for sample in test_samples]
    for model_type, model in models.items():
        perplexity = calculate_perplexity(model, tokenizers[model_type], eval_texts)
        metrics[model_type]["perplexity"] = perplexity
        print(f"{model_type} model perplexity: {perplexity:.4f}")
    
    # Generate responses and calculate other metrics
    print("\nGenerating responses and calculating other metrics...")
    for model_type, model in models.items():
        print(f"\n{model_type} model generation results:")
        print("-" * 80)
        
        for i, sample in enumerate(test_samples):
            instruction = sample["instruction"]
            input_text = sample["input"]
            
            response = generate_response(model, tokenizers[model_type], instruction, input_text, alignment, model_type)
            generated_responses[model_type].append(response)
            
            print(f"Test sample {i+1}: {instruction}")
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            print("-" * 40)
    
    # Calculate response length
    print("\nCalculating response length...")
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
        print(f"{model_type} model average response length: {avg_length:.2f} words")
        # 打印每个响应的实际长度
        for i, length in enumerate(lengths):
            print(f"  Test sample {i+1} response length: {length} words")
    
    # Calculate lexical diversity
    print("\nCalculating lexical diversity...")
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
        print(f"{model_type} model lexical diversity: {lexical_diversity:.4f}")
    
    # Calculate instruction following
    print("\nCalculating instruction following...")
    for model_type, responses in generated_responses.items():
        relevance_scores = []
        for i, (resp, sample) in enumerate(zip(responses, test_samples)):
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
        print(f"{model_type} model instruction following: {avg_relevance:.4f}")
    
    # 计算生成质量评估
    print("\nCalculating generation quality...")
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
        print(f"{model_type} model generation quality: {generation_quality:.4f}")
    
    # 保留response_quality字段以保持兼容性
    for model_type in metrics:
        if "generation_quality" in metrics[model_type]:
            metrics[model_type]["response_quality"] = metrics[model_type]["generation_quality"]
    
    return metrics, generated_responses

def plot_bar_chart(metrics, metric_name, title, save_path):
    """Plot bar chart"""
    model_types = list(metrics.keys())
    values = [metrics[model][metric_name] for model in model_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_types, values, color=['blue', 'green'])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to: {save_path}")

def plot_heatmap(metrics, save_path):
    """Plot heatmap"""
    model_types = list(metrics.keys())
    # 移除困惑度指标，只包含其他指标
    metric_names = [metric for metric in metrics[model_types[0]].keys() if metric != 'perplexity']
    
    # Prepare data
    data = []
    for model in model_types:
        row = [metrics[model][metric] for metric in metric_names]
        data.append(row)
    
    # Normalize data
    data = np.array(data)
    for i, metric in enumerate(metric_names):
        # 其他指标都是越高越好，不需要反转
        pass
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=metric_names,
                yticklabels=model_types)
    
    plt.title('Model Performance Heatmap (Excluding Perplexity)', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Metrics', fontsize=12)
    plt.ylabel('Model Types', fontsize=12)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {save_path}")


def plot_perplexity_comparison(metrics, save_path):
    """Plot perplexity comparison chart"""
    model_types = list(metrics.keys())
    perplexity_values = [metrics[model]['perplexity'] for model in model_types]
    
    plt.figure(figsize=(10, 6))
    
    # 创建柱状图
    bars = plt.bar(model_types, perplexity_values, color=['blue', 'green'])
    
    plt.title('Perplexity Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Model Types', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 添加值标签
    for bar, value in zip(bars, perplexity_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10000, f'{value:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Perplexity comparison chart saved to: {save_path}")

def plot_radar_chart(metrics, save_path):
    """Plot radar chart"""
    model_types = list(metrics.keys())
    # 排除困惑度和平均响应长度指标
    metric_names = [metric for metric in metrics[model_types[0]].keys() if metric not in ['perplexity', 'average_response_length']]
    num_metrics = len(metric_names)
    
    # Calculate angles
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # Prepare data
    colors = ['blue', 'green']
    for i, (model, color) in enumerate(zip(model_types, colors)):
        values = [metrics[model][metric] for metric in metric_names]
        
        # Normalize to 0-1 range
        values = np.array(values)
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)
        
        values = values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.25)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticklabels([])
    
    plt.title('Model Performance Radar Chart (Excluding Perplexity)', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to: {save_path}")

def plot_comparison_chart(metrics, save_path):
    """Plot comparison chart"""
    model_types = list(metrics.keys())
    # 排除困惑度指标，因为它已经有专门的图表
    metric_names = [metric for metric in metrics[model_types[0]].keys() if metric != 'perplexity']
    
    # Prepare data
    data = {metric: [metrics[model][metric] for model in model_types] for metric in metric_names}
    
    plt.figure(figsize=(14, 10))
    
    # 计算子图数量
    num_metrics = len(metric_names)
    rows = (num_metrics + 1) // 2  # 向上取整
    cols = 2
    
    for i, (metric, values) in enumerate(data.items(), 1):
        plt.subplot(rows, cols, i)
        bars = plt.bar(model_types, values, color=['blue', 'green'])
        plt.title(metric, fontsize=12, fontweight='bold')
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.4f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Performance Metrics Comparison (Excluding Perplexity)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to: {save_path}")

def main():
    """Main function"""
    # Test samples
    test_samples = [
        {"instruction": "什么是表闭水停证？", "input": ""},
        {"instruction": "什么是肺炎？", "input": ""},
        {"instruction": "请解释中医的阴阳理论", "input": ""},
    ]
    
    # Load knowledge base
    print("\nLoading knowledge base...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    
    # Load models
    models = {}
    tokenizers = {}
    
    model_types = ["baseline", "csmka"]
    for model_type in model_types:
        try:
            model, tokenizer = load_model(model_type)
            models[model_type] = model
            tokenizers[model_type] = tokenizer
            print(f"Successfully loaded {model_type} model")
        except Exception as e:
            print(f"Failed to load {model_type} model: {e}")
    
    # Calculate evaluation metrics
    metrics, generated_responses = calculate_metrics(models, tokenizers, test_samples, alignment)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "evaluation_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evaluation results
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Generate visualization images
    print("\nGenerating visualization images...")
    
    # Plot bar charts
    for metric_name in metrics[model_types[0]].keys():
        title = f"{metric_name} Comparison"
        save_path = os.path.join(output_dir, f"{metric_name}_bar.png")
        plot_bar_chart(metrics, metric_name, title, save_path)
    
    # Plot heatmap
    heatmap_path = os.path.join(output_dir, "metrics_heatmap.png")
    plot_heatmap(metrics, heatmap_path)
    
    # Plot perplexity comparison chart
    perplexity_path = os.path.join(output_dir, "perplexity_comparison.png")
    plot_perplexity_comparison(metrics, perplexity_path)
    
    # Plot radar chart
    radar_path = os.path.join(output_dir, "metrics_radar.png")
    plot_radar_chart(metrics, radar_path)
    
    # Plot comparison chart
    comparison_path = os.path.join(output_dir, "metrics_comparison.png")
    plot_comparison_chart(metrics, comparison_path)
    
    # Release memory
    for model in models.values():
        del model
    torch.cuda.empty_cache()
    
    print("\nEvaluation and visualization completed!")
    print(f"All images saved to: {output_dir}")

if __name__ == "__main__":
    main()
