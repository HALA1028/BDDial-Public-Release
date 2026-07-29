import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def process_txt_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    processed_lines = []
    for line in lines:
        # 删除行首的 "\n"
        line = line.lstrip("\\n")
        
        # 检查行中是否含有多余的换行符并处理
        if '\\n' in line:
            line = line.replace("\\n", "\n")  # 将所有的 \n 删除

        # 通用正则替换：将带内容或简单标签的分隔符修正
        # 第一种情况：匹配带内容的标签 [Hx. 内容, Hy. 内容]
        line = re.sub(
            r'\[(H\d+\.\s*[^\],]+)\s*[,\s，、]?\s*(H\d+\.\s*[^\]]+)\]',
            r'[\1][\2]',
            line
        )
        
        # 第二种情况：匹配简单标签 [H1 H2]
        line = re.sub(
            r'\[(H\d+)\s*[,\s\u3001]+\s*(H\d+)]',
            lambda m: ''.join(f'[{tag}]' for tag in m.group(0)[1:-1].replace(',', ' ').split()),
            line
        )
        
        # 添加处理后的行
        processed_lines.append(line)
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(processed_lines)

def process_all_txt_files(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有txt文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            process_txt_file(input_file_path, output_file_path)



# Example usage
input_folder = os.path.join(BASE_DIR, 'stage2_formal_output')
output_folder = os.path.join(BASE_DIR, 'stage2_formal_clean_output')
process_all_txt_files(input_folder, output_folder)
