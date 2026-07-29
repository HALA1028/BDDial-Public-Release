import os
import json
from tqdm import tqdm
from openai import OpenAI
client = OpenAI()

# --- 配置区 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(BASE_DIR, "HY_single_turn_access", "output_jsons_completed_copy")
mapping_file = os.path.join(BASE_DIR, "HY_rules", "HY_rules.json")
output_folder = os.path.join(BASE_DIR, "HY_single_turn_access", "output_jsons_completed_fixed")
model_name = "gpt-4o"
# --- 配置区结束 ---
 
# 读取寄送档案
with open(mapping_file, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

# 建立 id 到 options 映射
id_to_options = {}
for item in mapping_data['items']:
    id_to_options[item['id']] = item['options']

# 确保输出资料夹存在
os.makedirs(output_folder, exist_ok=True)

def generate_response(n_text, standard_text):
    """调用 OpenAI API，生成简短自然的病人回答"""
    prompt = f"""
                你是一个双向情感障碍症的患者，你需要根据标准描述对N发言进行回复
                根据下面的信息，以聊天的口吻，简短回答（不超过20字）你会怎么回应。

                N发言是：
                "{n_text}"

                你的回话，需要依照的标准描述是：
                "{standard_text}"

                请根据标准描述和提问内容，推测一个简短自然的回答。
                注意：用自然、不正式的口吻！

                直接给出回答，不要解释。
                    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个擅长心理咨询对话模拟的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=64,
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        print(f"调用API时出错: {e}")
        return None

# 遍历所有子目录和文件
for root, dirs, files in os.walk(input_folder):
    for file in tqdm(files, desc=f"处理 {root}"):
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            output_path_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_path_dir, exist_ok=True)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            correct_score = os.path.splitext(file)[0]  # 文件名就是正确的score

            new_data = []
            for item in data:
                n_text = item['N']
                #item_id = item.get('id', None)
                item_id = os.path.basename(relative_path)

                if not item_id or item_id not in id_to_options:
                    print(f"警告：找不到 id {item_id} 的映射，跳过")
                    continue

                # 找到正确score对应的标准text
                options = id_to_options[item_id]
                standard_text = None
                for opt in options:
                    if str(opt['score']) == correct_score:
                        standard_text = opt['text']
                        break

                if standard_text is None:
                    print(f"警告：id {item_id} 找不到score {correct_score}的描述，跳过")
                    continue

                # 调用GPT生成P
                generated_p = generate_response(n_text, standard_text)

                new_item = {
                    "N": n_text,
                    "P": generated_p if generated_p else item['P'],  # 如果API失败就用原本的P
                    "score": correct_score
                }
                if 'id' in item:  # 保留id
                    new_item['id'] = item['id']

                new_data.append(new_item)

            # 保存新文件
            output_file_path = os.path.join(output_path_dir, file)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

print("全部处理完成 ✅")
