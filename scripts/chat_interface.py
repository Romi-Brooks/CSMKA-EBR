import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_chat_model():
    """加载训练好的对话模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载基础模型和LoRA适配器
    base_model_path = os.path.join(base_dir, "models", "Qwen3-1.7B")
    adapter_path = os.path.join(base_dir, "output", "lora_model", "checkpoint-63")
    
    print("正在加载模型...")
    print(f"基础模型路径: {base_model_path}")
    print(f"LoRA适配器路径: {adapter_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        dtype=torch.float16
    )
    
    # 设置为评估模式
    model.eval()
    
    print("模型加载完成！")
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text=""):
    """生成模型回复"""
    # 构建提示词
    prompt = f"### Instruction:\n请你必须使用中文回答以下问题，不要使用任何英文: {instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:\n"
    
    # 生成回复
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:\n")[-1].strip()
    
    return response

def chat_interface():
    """对话界面"""
    # 加载模型
    try:
        model, tokenizer = load_chat_model()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    print("\n==========================================")
    print("          中医医学对话助手")
    print("==========================================")
    print("你可以输入任何医学相关问题，按Enter发送")
    print("输入 '退出' 或 'exit' 结束对话")
    print("==========================================")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ")
            
            # 检查是否退出
            if user_input.strip().lower() in ["退出", "exit"]:
                print("感谢使用，再见！")
                break
            
            # 生成回复
            print("助手: ", end="")
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"生成回复时出错: {e}")
            print("请尝试重新输入问题")

if __name__ == "__main__":
    chat_interface()
