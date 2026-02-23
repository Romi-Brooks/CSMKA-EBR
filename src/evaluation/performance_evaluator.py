import nltk
from rouge_score import rouge_scorer
import numpy as np

Path = 'C:\\Users\\romic\\AppData\\Roaming\\nltk_data'

nltk.download('punkt_tab', download_dir=Path, quiet=True)
nltk.download('punkt', download_dir=Path, quiet=True)
nltk.download('wordnet', download_dir=Path, quiet=True)

class PerformanceEvaluator:
    def __init__(self):
        """初始化评估器"""
        nltk.download('punkt')
        nltk.download('wordnet')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate(self, predictions, references):
        """评估模型性能"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考长度不匹配")
        
        metrics = {
            "relevance": [],  # 相关性评分
            "coherence": [],  # 连贯性评分
            "medical_accuracy": []  # 医学准确性评分
        }
        
        # 医学关键词列表
        medical_keywords = [
            "肺炎", "高血压", "糖尿病", "心脏病", "感冒", "流感",
            "症状", "治疗", "预防", "病因", "方法", "药物"
        ]
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            print(f"样本 {i+1}:")
            print(f"预测: {pred[:100]}...")
            print(f"参考: {ref[:100]}...")
            
            # 计算相关性：预测中包含的医学关键词数量
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # 计算关键词匹配度
            pred_keywords = set()
            for keyword in medical_keywords:
                if keyword in pred_lower:
                    pred_keywords.add(keyword)
            
            ref_keywords = set()
            for keyword in medical_keywords:
                if keyword in ref_lower:
                    ref_keywords.add(keyword)
            
            # 计算相关性分数
            if ref_keywords:
                relevance = len(pred_keywords.intersection(ref_keywords)) / len(ref_keywords)
            else:
                relevance = 0
            
            # 计算连贯性：回答长度和完整性
            coherence = min(len(pred) / 50, 1.0) if len(pred) > 0 else 0
            
            # 计算医学准确性：包含医学关键词的比例
            medical_accuracy = len(pred_keywords) / len(medical_keywords) if medical_keywords else 0
            
            metrics["relevance"].append(relevance)
            metrics["coherence"].append(coherence)
            metrics["medical_accuracy"].append(medical_accuracy)
            
            print(f"相关性: {relevance:.4f}")
            print(f"连贯性: {coherence:.4f}")
            print(f"医学准确性: {medical_accuracy:.4f}")
        
        # 计算平均值
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics

    def compare_models(self, model1_results, model2_results):
        """比较两个模型的性能"""
        comparison = {}
        for metric in model1_results:
            if metric in model2_results:
                comparison[metric] = {
                    "model1": model1_results[metric],
                    "model2": model2_results[metric],
                    "improvement": model2_results[metric] - model1_results[metric]
                }
        
        return comparison
    
    def evaluate_with_evidence(self, model_outputs, references):
        """评估包含证据的模型输出"""
        predictions = [output.get("answer", "") for output in model_outputs]
        return self.evaluate(predictions, references)
    
    def get_comprehensive_report(self, model_name, metrics, baseline_metrics=None):
        """生成综合评估报告"""
        report = f"# {model_name} 性能评估报告\n\n"
        
        # 详细指标
        report += "## 详细指标\n"
        for metric, value in metrics.items():
            report += f"- {metric}: {value:.4f}\n"
        report += "\n"
        
        # 与基线模型对比
        if baseline_metrics:
            report += "## 与基线模型对比\n"
            for metric in metrics:
                if metric in baseline_metrics:
                    improvement = metrics[metric] - baseline_metrics[metric]
                    report += f"- {metric}: {metrics[metric]:.4f} (vs {baseline_metrics[metric]:.4f}, {'+' if improvement > 0 else ''}{improvement:.4f})\n"
            report += "\n"
        
        # 总结
        report += "## 总结\n"
        if baseline_metrics:
            improved_metrics = [m for m in metrics if metrics[m] > baseline_metrics.get(m, 0)]
            report += f"模型在 {len(improved_metrics)} 个指标上优于基线模型\n"
        else:
            report += "模型性能评估完成\n"
        
        return report
    
    def evaluate_f1_score(self, predictions, references):
        """计算F1分数"""
        from sklearn.metrics import f1_score
        
        # 简单的F1计算（基于关键词）
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_set = set(pred.lower().split())
            ref_set = set(ref.lower().split())
            
            if not pred_set and not ref_set:
                f1 = 1.0
            elif not pred_set or not ref_set:
                f1 = 0.0
            else:
                tp = len(pred_set.intersection(ref_set))
                fp = len(pred_set - ref_set)
                fn = len(ref_set - pred_set)
                
                if tp + fp + fn == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * tp / (2 * tp + fp + fn)
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores)