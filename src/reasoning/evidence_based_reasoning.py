# src/reasoning/evidence_based_reasoning.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

class EvidenceBasedReasoning:
    def __init__(self, model_name_or_path, lora_path=None):
        """初始化推理模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if lora_path:
            # 加载LoRA微调模型
            try:
                config = PeftConfig.from_pretrained(lora_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=False
                )
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            except Exception as e:
                print(f"LoRA模型加载失败: {e}")
                print("使用基础模型...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=False
                )
        else:
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False
            )
        
        # 设置模型为评估模式
        self.model.eval()
    
    def generate_answer(self, prompt, max_new_tokens=100, temperature=0.5):
        """生成回答"""
        # 确保提示词清晰明确
        if not prompt:
            prompt = "请提供医学专业知识"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 将输入数据移到模型所在的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.2  # 添加重复惩罚，减少重复内容
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取回答部分
            if "回答:" in answer:
                answer = answer.split("回答:")[-1].strip()
            
            return answer if answer else "未生成有效的回答"
        except Exception as e:
            print(f"生成回答失败: {e}")
            return "生成回答时出错"
    
    def reason_with_evidence(self, question, evidence, max_new_tokens=200):
        """基于多模态证据进行推理"""
        # 构建提示词
        prompt = self._build_reasoning_prompt(question, evidence)
        
        # 生成回答
        answer = self.generate_answer(prompt, max_new_tokens=max_new_tokens)
        
        return {
            "question": question,
            "evidence": evidence,
            "answer": answer,
            "prompt": prompt
        }
    
    def _build_reasoning_prompt(self, question, evidence):
        """构建推理提示词"""
        prompt = f"请用中文回答以下医学问题，回答要专业、准确、简洁。\n\n"
        prompt += f"问题: {question}\n\n"
        
        # 添加文本证据
        if evidence.get("text_evidence"):
            prompt += "相关医学知识:\n"
            for i, item in enumerate(evidence["text_evidence"][:2]):
                prompt += f"{i+1}. {item['content']}\n"
            prompt += "\n"
        
        prompt += "回答:\n"
        return prompt