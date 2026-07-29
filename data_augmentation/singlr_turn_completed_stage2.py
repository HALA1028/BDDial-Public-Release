# File: augment_and_copy.py

import os
import json
import random
import time
from openai import OpenAI

client = OpenAI()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(ROOT_DIR, "HY_single_turn_access", "fixed_merged_augmented_augmented")
NEW_PARENT_DIR = BASE_DIR + "_augmented"
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.8

def augment_file(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as f:
        turns = json.load(f)

    n_orig = len(turns)
    new_turns = []

    if n_orig < 50:
        samples = turns
    else:
        needed = max(0, 100 - n_orig)
        samples = random.sample(turns, needed)

    for idx, turn in enumerate(samples, 1):
        if not turn.get('N') or not turn['N'].strip():
            continue

        prompt = (
            f"请根据下面的用户发言，用不同的风格重新表述，保持含义一致，仅输出新的用户提问文本，不要包含引号或多余说明。\n"
            f"原始用户发言：{turn['N']}"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "你是一个善于修改对话风格的专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            new_text = response.choices[0].message.content
            new_turn = {
                "N": new_text,
                "P": turn['P'],
                "score": turn['score']
            }
            new_turns.append(new_turn)

        except Exception as e:
            print(f"Error processing turn {idx} in {file_path}: {e}")
        time.sleep(1)

    augmented = turns + new_turns

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    base_no_ext, ext = os.path.splitext(base_name)
    out_aug_path = os.path.join(output_dir, f"{base_no_ext}_aug{ext}")

    with open(out_aug_path, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

def traverse_and_augment(base_dir, new_base_dir):
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith('.json'):
                file_path = os.path.join(root, fname)
                rel_path = os.path.relpath(root, base_dir)
                output_dir = os.path.join(new_base_dir, rel_path)
                print(f"Processing {file_path}...")
                augment_file(file_path, output_dir)

if __name__ == '__main__':
    for subfolder in os.listdir(BASE_DIR):
        subfolder_path = os.path.join(BASE_DIR, subfolder)
        if os.path.isdir(subfolder_path):
            new_subfolder_path = os.path.join(NEW_PARENT_DIR, subfolder)
            os.makedirs(new_subfolder_path, exist_ok=True)
            traverse_and_augment(subfolder_path, new_subfolder_path)
    print("All files processed and augmented into:", NEW_PARENT_DIR)
