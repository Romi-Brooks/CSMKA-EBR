import os
import json
import numpy as np
import faiss
from PIL import Image
import clip
import torch

class KnowledgeBaseBuilder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.vector_dim = 512  # CLIP ViT-B/32 的特征维度
        self.index = faiss.IndexFlatIP(self.vector_dim)  # 内积索引
        self.metadata = []  # 存储元数据：类型、路径、描述等
    
    def add_image(self, image_path, description=""):
        """添加医学图像到知识库"""
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image).cpu().numpy()
            
            # 归一化特征
            image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            
            # 添加到向量库
            self.index.add(image_features)
            self.metadata.append({"type": "image", "path": image_path, "description": description})
            return True
        except Exception as e:
            print(f"添加图像失败 {image_path}: {e}")
            return False
    
    def add_text(self, text):
        """添加医学文本到知识库"""
        try:
            # 处理中医药文本的特殊结构
            processed_text = text
            
            # 提取章节信息
            sections = []
            
            # 分割章节
            lines = processed_text.split('\n')
            current_section = {"title": "", "content": []}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 处理目录标记
                if "<目录>" in line:
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"title": line, "content": []}
                # 处理篇名标记
                elif "<篇名>" in line:
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"title": line, "content": []}
                # 处理内容
                elif line.startswith("内容："):
                    content_part = line[3:].strip()
                    if content_part:
                        current_section["content"].append(content_part)
                else:
                    if line:
                        current_section["content"].append(line)
            
            # 添加最后一个章节
            if current_section["content"]:
                sections.append(current_section)
            
            # 处理章节内容
            short_sentences = []
            for section in sections:
                # 组合章节标题和内容
                section_text = section["title"] + " " + " ".join(section["content"])
                
                # 分割长文本为适合CLIP的短句
                # 首先按句号分割
                sentences = section_text.split('。')
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # 检查句子长度，确保不超过CLIP的token限制
                    # CLIP限制为77个tokens，中文大约可以容纳30-40个汉字
                    if len(sentence) > 40:
                        # 智能分割长句子
                        # 按逗号分割
                        comma_parts = sentence.split('，')
                        if len(comma_parts) > 1:
                            for part in comma_parts:
                                part = part.strip()
                                if part and len(part) > 5:
                                    short_sentences.append(part)
                        else:
                            # 按长度分割
                            for i in range(0, len(sentence), 40):
                                chunk = sentence[i:i+40].strip()
                                if chunk and len(chunk) > 5:
                                    short_sentences.append(chunk)
                    else:
                        if len(sentence) > 5:
                            short_sentences.append(sentence)
            
            # 添加短句子到知识库
            added = 0
            # 处理整个文件的所有句子
            for sentence in short_sentences:
                try:
                    # 确保句子能够被CLIP正确处理
                    # 递归分割句子直到能够被CLIP处理
                    def process_sentence(s):
                        try:
                            # 尝试tokenize
                            text_input = clip.tokenize([s]).to(self.device)
                            with torch.no_grad():
                                text_features = self.model.encode_text(text_input).cpu().numpy()
                            
                            # 归一化特征
                            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
                            
                            # 添加到向量库
                            self.index.add(text_features)
                            self.metadata.append({"type": "text", "content": s})
                            nonlocal added
                            added += 1
                            print(f"成功添加句子: {s[:100]}...")
                            return True
                        except Exception as e:
                            # 如果句子太长，进一步分割
                            if "too long for context length" in str(e):
                                # 按更小的粒度分割
                                if len(s) > 20:
                                    # 按20个字符分割
                                    for i in range(0, len(s), 20):
                                        chunk = s[i:i+20].strip()
                                        if chunk and len(chunk) > 5:
                                            process_sentence(chunk)
                                return False
                            else:
                                print(f"添加句子失败: {e}")
                                return False
                    
                    # 处理句子
                    process_sentence(sentence)
                except Exception as e:
                    print(f"添加句子失败: {e}")
            
            return added > 0
        except Exception as e:
            print(f"添加文本失败: {e}")
            return False
        
    def build_from_directory(self, image_dir, text_dir):
        """从目录批量构建知识库"""
        print(f"=== 开始构建知识库 ===")
        print(f"图像目录: {image_dir}")
        print(f"文本目录: {text_dir}")
        
        # 确保目录存在
        if not os.path.exists(image_dir):
            print(f"创建图像目录: {image_dir}")
            os.makedirs(image_dir, exist_ok=True)
        
        if not os.path.exists(text_dir):
            print(f"创建文本目录: {text_dir}")
            os.makedirs(text_dir, exist_ok=True)
        
        # 添加图像
        image_count = 0
        if os.path.exists(image_dir):
            print(f"开始处理图像，目录包含 {len(os.listdir(image_dir))} 个文件")
            for img_file in os.listdir(image_dir):
                if img_file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(image_dir, img_file)
                    print(f"处理图像: {img_file}")
                    if self.add_image(img_path, description=img_file):
                        image_count += 1
            print(f"成功添加 {image_count} 个图像")
        # 添加文本
        if os.path.exists(text_dir):
            for txt_file in os.listdir(text_dir):
                if txt_file.endswith(".txt"):
                    txt_path = os.path.join(text_dir, txt_file)
                    print(f"\n=== 处理文件: {txt_file} ===")
                    print(f"文件路径: {txt_path}")
                    print(f"文件存在: {os.path.exists(txt_path)}")
                    print(f"文件大小: {os.path.getsize(txt_path)} 字节")
                    
                    try:
                        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        print(f"读取成功，内容长度: {len(content)} 字符")
                        
                        # 处理整个文件内容
                        if len(content) > 20:  # 至少20个字符
                            # 不限制长度，让add_text方法处理分割
                            success = self.add_text(content)
                            print(f"添加结果: {'成功' if success else '失败'}")
                        else:
                            print("内容太短，跳过")
                            
                    except Exception as e:
                        print(f"错误: {e}")
                        import traceback
                        traceback.print_exc()
                        
    def save(self, index_path, metadata_path):
        """保存知识库"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def load(self, index_path, metadata_path):
        """加载知识库"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)