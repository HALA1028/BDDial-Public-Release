import os
from openai import OpenAI
from tqdm import tqdm  # 新增导入
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 固定的 system prompt
stage1_sys_prompt = os.path.join(BASE_DIR, "prompt", "stage1_sys_prompt.txt")
with open(stage1_sys_prompt, 'r', encoding='utf-8') as file:
            SYSTEM_PROMPT = file.read()

# 输入文件夹和输出文件夹
input_folder = os.path.join(BASE_DIR, "stage1_formal_v2")
output_folder = os.path.join(BASE_DIR, "stage1_formal_output_v2")
os.makedirs(output_folder, exist_ok=True)

# 获取所有需要处理的 txt 文件
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# 处理每个文件
for filename in tqdm(txt_files, desc="处理进度", unit="文件"):
    input_file_path = os.path.join(input_folder, filename)
        
    # 读取用户的 prompt（txt 文件内容）
    with open(input_file_path, 'r', encoding='utf-8') as file:
        user_prompt = file.read()

    # 调用 GPT-4 API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt+'Your HY items-related questions is:'},
            ],
        )
        # 获取生成的文本
        assistant_response = response.choices[0].message.content

        # 将结果写入到输出文件夹
        output_file_path = os.path.join(output_folder, f"processed_{filename}")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(assistant_response)

        print(f"已处理文件: {filename}")

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")

print(f"所有文件已处理，结果保存在文件夹 '{output_folder}' 中。")
