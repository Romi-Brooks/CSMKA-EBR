import numpy as np
import clip
import torch

class KnowledgeAlignment:
    def __init__(self, knowledge_base, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.knowledge_base = knowledge_base
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
    
    def query(self, question, top_k=5):
        """查询多模态知识库，返回最相关的证据"""
        # 编码问题
        text_input = clip.tokenize([question]).to(self.device)
        with torch.no_grad():
            query_features = self.model.encode_text(text_input).cpu().numpy()
        
        # 归一化特征
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        
        # 在知识库中检索
        distances, indices = self.knowledge_base.index.search(query_features, top_k)
        
        # 收集证据
        evidence = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.knowledge_base.metadata):
                item = self.knowledge_base.metadata[idx]
                evidence_item = {
                    "type": item["type"],
                    "score": float(distances[0][i]),
                    "content": item.get("content", item.get("path", ""))
                }
                evidence.append(evidence_item)
        
        return evidence
    
    def get_multimodal_evidence(self, question, top_k_text=3, top_k_image=2):
        """获取多模态证据，包括文本和图像"""
        all_evidence = self.query(question, top_k=top_k_text + top_k_image)
        
        # 分离文本和图像证据
        text_evidence = [e for e in all_evidence if e["type"] == "text"][:top_k_text]
        image_evidence = [e for e in all_evidence if e["type"] == "image"][:top_k_image]
        
        return {
            "text_evidence": text_evidence,
            "image_evidence": image_evidence,
            "all_evidence": all_evidence
        }
    
    def build_prompt_with_evidence(self, question, evidence):
        """构建包含证据的提示词"""
        prompt = f"问题: {question}\n\n"
        prompt += "相关医学知识:\n"
        
        # 添加文本证据
        text_evidence = evidence.get("text_evidence", [])
        for i, item in enumerate(text_evidence[:3]):
            prompt += f"{i+1}. {item['content']}\n"
        
        # 添加图像证据描述
        image_evidence = evidence.get("image_evidence", [])
        if image_evidence:
            prompt += "\n相关医学图像:\n"
            for i, item in enumerate(image_evidence[:2]):
                prompt += f"{i+1}. {item['content']}\n"
        
        prompt += "\n请基于上述医学知识，提供专业、准确的回答:\n"
        
        return prompt