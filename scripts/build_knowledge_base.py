import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base.build_knowledge_base import KnowledgeBaseBuilder

if __name__ == "__main__":
    # 获取脚本所在目录的父目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    image_dir = os.path.join(data_dir, "images")
    text_dir = os.path.join(data_dir, "texts")

    print(f"基础目录: {base_dir}")
    print(f"数据目录: {data_dir}")
    print(f"图像目录: {image_dir}")
    print(f"文本目录: {text_dir}")
    
    # 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "database")
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "knowledge_base.index")
    metadata_path = os.path.join(output_dir, "knowledge_base_metadata.json")
    
    print("开始构建多模态知识库...")
    print(f"图像目录: {image_dir}")
    print(f"文本目录: {text_dir}")
    
    # 初始化知识库构建器
    builder = KnowledgeBaseBuilder()
    
    # 从目录构建知识库
    builder.build_from_directory(image_dir, text_dir)
    
    # 保存知识库
    builder.save(index_path, metadata_path)
    
    print(f"知识库构建完成！")
    print(f"向量库保存路径: {index_path}")
    print(f"元数据保存路径: {metadata_path}")
    print(f"知识库大小: {len(builder.metadata)} 条记录")
    
    # 统计信息
    image_count = sum(1 for item in builder.metadata if item["type"] == "image")
    text_count = sum(1 for item in builder.metadata if item["type"] == "text")
    
    print(f"- 图像数量: {image_count}")
    print(f"- 文本数量: {text_count}")