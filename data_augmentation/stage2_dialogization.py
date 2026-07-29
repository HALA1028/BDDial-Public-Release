import os
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 固定的 system prompt
stage2_sys_prompt = os.path.join(BASE_DIR, "prompt", "stage2_sys_prompt_new.txt")
with open(stage2_sys_prompt, 'r', encoding='utf-8') as file:
    SYSTEM_PROMPT = file.read()

# 输入文件夹和输出文件夹
input_folder = os.path.join(BASE_DIR, "stage2_formal_v2_selected")
output_folder = os.path.join(BASE_DIR, "stage2_formal_output_v2_selected")
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .txt 文件名
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# 处理每个文件，添加 tqdm 进度条
for filename in tqdm(txt_files, desc="Processing files", unit="file"):
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
                {"role": "user", "content": user_prompt + 'Your rewritten multi-turn dialog is:'},
            ],
        )
        # 获取生成的文本
        assistant_response = response.choices[0].message.content

        # 将结果写入到输出文件夹
        output_file_path = os.path.join(output_folder, f"processed_{filename}")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(assistant_response)

    except Exception as e:
        print(f"\n处理文件 {filename} 时出错: {e}")

print(f"\n所有文件已处理，结果保存在文件夹 '{output_folder}' 中。")
