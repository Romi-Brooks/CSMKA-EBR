import os
import chardet
# 该脚本用于转换所有非UTF-8的数据集文件到UTF-8，仅用于txt
def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def convert_to_utf8(input_path, output_path):
    """将文件转换为UTF-8编码"""
    encoding = detect_encoding(input_path)
    print(f"检测到编码: {encoding} for {os.path.basename(input_path)}")
    
    try:
        with open(input_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def process_directory(input_dir):
    """处理目录中的所有文本文件"""
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(input_dir, f"{filename}")
            
            print(f"处理文件: {filename}")
            success = convert_to_utf8(input_path, output_path)
            if success:
                print(f"转换成功: {filename}")
            else:
                print(f"转换失败: {filename}")

if __name__ == "__main__":
    text_dir = "data/texts"
    print(f"开始处理目录: {text_dir}")
    process_directory(text_dir)
    print("处理完成!")