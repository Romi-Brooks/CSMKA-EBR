import os
import sys

# 配置 Hugging Face 镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fine_tuning.lora_fine_tuning import LoRAFineTuner

if __name__ == "__main__":
    # 数据目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    print(f"正确的数据目录: {data_dir}")
    print(f"目录是否存在: {os.path.exists(data_dir)}")
    
    # 输出目录
    output_dir = "../output/lora_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始微调模型...")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 初始化微调器 - 使用绝对路径加载 Qwen3-1.7B 模型
    model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
    print(f"模型路径: {model_path}")
    print(f"模型目录是否存在: {os.path.exists(model_path)}")
    tuner = LoRAFineTuner(base_model_name=model_path)
    
    # 执行微调
    try:
        result_dir = tuner.fine_tune(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=2,
            batch_size=4
        )
        
        print(f"模型微调完成！")
        print(f"微调模型保存路径: {result_dir}")
        
    except Exception as e:
        print(f"微调过程中出错: {e}")
        print("如果内存不足，可以尝试使用更小的模型或调整batch_size")
        print("例如：可以使用 'facebook/opt-1.3b' 作为基础模型")
        
        # 提供备选方案
        print("\n备选方案：使用更小的batch size")
        try:
            tuner = LoRAFineTuner(base_model_name=model_path)
            result_dir = tuner.fine_tune(
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=2,
                batch_size=2
            )
            print(f"使用备选batch size微调完成！")
            print(f"微调模型保存路径: {result_dir}")
        except Exception as e2:
            print(f"备选方案也失败: {e2}")
            print("请检查硬件资源或尝试使用更小规模的模型")