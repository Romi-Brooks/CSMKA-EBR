from transformers import AutoConfig

# 只加载模型配置
print("正在加载模型配置...")
config = AutoConfig.from_pretrained('./models/Qwen-7B-Chat', trust_remote_code=True)

print("\n模型配置:")
print(config)

# 查看模型类型
print(f"\n模型类型: {config.model_type}")

# 对于Qwen模型，通常的LoRA目标模块是
print("\nQwen模型推荐的LoRA目标模块:")
print(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

# 尝试直接查看modeling_qwen.py文件
print("\n尝试查看modeling_qwen.py文件中的模块定义...")
try:
    with open('./models/Qwen-7B-Chat/modeling_qwen.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 查找QWenAttention类
        if 'class QWenAttention' in content:
            print("\n=== QWenAttention类 ===")
            start_idx = content.find('class QWenAttention')
            end_idx = content.find('class ', start_idx + 1)
            if end_idx == -1:
                end_idx = len(content)
            attention_class = content[start_idx:end_idx]
            print(attention_class[:2000])  # 只打印前2000个字符
        
        # 查找QWenMLP类
        if 'class QWenMLP' in content:
            print("\n=== QWenMLP类 ===")
            start_idx = content.find('class QWenMLP')
            end_idx = content.find('class ', start_idx + 1)
            if end_idx == -1:
                end_idx = len(content)
            mlp_class = content[start_idx:end_idx]
            print(mlp_class[:2000])  # 只打印前2000个字符
            
except Exception as e:
    print(f"查看文件失败: {e}")