import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

class LoRAFineTuner:
    def __init__(self, base_model_name="./models/Qwen3-1.7B"):
        """初始化LoRA微调器"""
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.peft_model = None
    
    def load_data(self, data_dir, max_samples=1000):
        """加载数据集，限制样本数量"""
        data = []
        
        print(f"开始加载数据，数据目录: {data_dir}")
        
        # 确保目录存在
        if not os.path.exists(data_dir):
            print(f"数据目录不存在: {data_dir}")
            # 创建示例数据
            data.append({
                "instruction": "什么是肺炎？",
                "input": "",
                "output": "肺炎是肺部的炎症，通常由细菌、病毒或真菌引起。"
            })
            return Dataset.from_list(data)
        
        # 加载txt文件
        txt_dir = os.path.join(data_dir, "texts")
        if os.path.exists(txt_dir):
            txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
            print(f"找到 {len(txt_files)} 个文本文件")
            for txt_file in txt_files[:5]:  # 只处理前5个文本文件
                try:
                    with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if len(content) > 100:
                        data.append({
                            "instruction": "请总结以下医学内容",
                            "input": content[:300],  # 限制输入长度
                            "output": "医学专业内容摘要"
                        })
                        print(f"成功加载文本文件: {txt_file}")
                        if len(data) >= max_samples:
                            break
                except Exception as e:
                    print(f"读取txt文件失败 {txt_file}: {e}")
        
        # 加载json对话数据
        dialogues_dir = os.path.join(data_dir, "dialogues")
        if os.path.exists(dialogues_dir) and len(data) < max_samples:
            json_files = [f for f in os.listdir(dialogues_dir) if f.endswith(".json")]
            print(f"找到 {len(json_files)} 个对话文件")
            for json_file in json_files:
                try:
                    json_path = os.path.join(dialogues_dir, json_file)
                    print(f"尝试加载对话文件: {json_file}")
                    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        loaded = 0
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                dialogue = json.loads(line)
                                if isinstance(dialogue, dict) and "query" in dialogue and "response" in dialogue:
                                    data.append({
                                        "instruction": dialogue["query"],
                                        "input": "",
                                        "output": dialogue["response"]
                                    })
                                    loaded += 1
                                    if len(data) >= max_samples:
                                        break
                            except json.JSONDecodeError:
                                continue
                    print(f"成功加载 {loaded} 条对话数据")
                    if len(data) >= max_samples:
                        break
                except Exception as e:
                    print(f"读取json文件失败 {json_file}: {e}")
        
        # 确保至少有一些数据
        if len(data) == 0:
            data.append({
                "instruction": "什么是肺炎？",
                "input": "",
                "output": "肺炎是肺部的炎症，通常由细菌、病毒或真菌引起。"
            })
        
        print(f"加载完成，共 {len(data)} 条数据")
        dataset = Dataset.from_list(data)
        return dataset

    def preprocess_function(self, examples):
        """预处理函数"""
        # 确保指令和输出是字符串
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples.get("output", [""] * len(instructions))
        
        # 确保所有值都是字符串
        instructions = [str(instr) if instr is not None else "" for instr in instructions]
        inputs = [str(inp) if inp is not None else "" for inp in inputs]
        outputs = [str(out) if out is not None else "" for out in outputs]
        
        # 构建提示词
        prompts = []
        for instr, inp, out in zip(instructions, inputs, outputs):
            prompt = f"### Instruction:\n{instr}\n"
            if inp:
                prompt += f"### Input:\n{inp}\n"
            prompt += f"### Response:\n{out}\n"
            prompts.append(prompt)
        
        # 分词，确保设置正确的参数
        tokenized = self.tokenizer(
            prompts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 确保labels正确设置
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized

    def setup_lora(self, r=8, lora_alpha=16, lora_dropout=0.1):
        """设置LoRA配置"""
        # 为Qwen3-1.7B模型设置正确的LoRA目标模块
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 明确指定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        try:
            # 加载本地Qwen3-1.7B模型
            print(f"正在加载模型: {self.base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
    def fine_tune(self, data_dir, output_dir="./output", epochs=5, batch_size=4):
        """执行微调"""
        # 加载数据
        dataset = self.load_data(data_dir, max_samples=1000)
        
        # 预处理数据
        try:
            tokenized_dataset = dataset.map(
                self.preprocess_function, 
                batched=True,
                remove_columns=dataset.column_names  # 移除原始列
            )
            print(f"数据预处理完成，样本数: {len(tokenized_dataset)}")
        except Exception as e:
            print(f"数据预处理失败: {e}")
            # 创建一个简单的示例数据集
            sample_data = [{
                "instruction": "什么是肺炎？",
                "input": "",
                "output": "肺炎是肺部的炎症，通常由细菌、病毒或真菌引起。"
            }]
            dataset = Dataset.from_list(sample_data)
            tokenized_dataset = dataset.map(
                self.preprocess_function, 
                batched=True,
                remove_columns=dataset.column_names
            )
        # 设置LoRA
        self.setup_lora()
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"),
            logging_steps=1,
            save_strategy="epoch",
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            fp16=True,
            gradient_accumulation_steps=4,
            remove_unused_columns=False,
            dataloader_pin_memory=torch.cuda.is_available()
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return output_dir
    
    def evaluate(self, eval_dataset):
        """评估模型"""
        if not self.peft_model:
            raise ValueError("模型未初始化")
        
        # 这里可以添加评估逻辑
        # 例如计算损失、准确率等
        pass