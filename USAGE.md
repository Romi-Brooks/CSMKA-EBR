# 使用说明

本项目实现了跨体系多模态知识对齐与循证推理算法（CSMKA-EBR），使用CLIP进行多模态知识对齐，将医学图像、文本描述和专业术语映射到同一语义空间，实现跨模态知识检索与循证推理。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

- **医学文本**：已在 `data/texts/medical_knowledge.txt` 中提供示例数据
- **医学图像**：可在 `data/images/` 目录中添加医学相关图像（如CT、X光、药品包装等）
- **对话数据**：可在 `data/dialogues/` 目录中添加JSON格式的对话数据

### 3. 构建多模态知识库

```bash
python scripts/build_knowledge_base.py
```

该脚本会：
- 使用CLIP提取医学图像的特征
- 使用CLIP提取医学文本的特征
- 将所有特征存入FAISS向量库
- 生成知识库索引和元数据

### 4. 微调模型（可选）

```bash
python scripts/fine_tune_model.py
```

该脚本会：
- 使用LoRA微调对话模型
- 处理txt和json格式的训练数据
- 保存微调后的模型

### 5. 运行交互式推理

```bash
python scripts/run_inference.py
```

输入医学问题，系统会：
- 检索相关的多模态证据（文本和图像）
- 基于证据生成专业的回答
- 显示检索到的证据和生成的回答

### 6. 评估模型

```bash
python scripts/test_csmka.py
```

测试模型训练情况


## 验证需求可行性

要验证CSMKA-EBR算法的可行性，只需运行综合评估脚本：

```bash
python scripts/comprehensive_evaluation.py
```

### 预期结果

评估脚本会生成一个综合报告，显示两个模型的性能指标：
1. **baseline模型**：未经过微调的原始模型
2. **CSMKA模型**：使用知识库的模型


### 可视化验证

如果需要更直观的验证，可以运行可视化评估脚本：

```bash
python scripts/full_evaluation_img.py
```

该脚本会生成多种可视化图表，包括：
- 各指标的柱状图
- 模型性能热图（不含困惑度）
- 专门的困惑度比较图表
- 模型性能雷达图

所有图表会保存到 `output/evaluation_images/` 目录中，便于直观分析模型性能差异。
