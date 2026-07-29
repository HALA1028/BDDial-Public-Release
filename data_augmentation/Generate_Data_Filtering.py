#本code用作判断生成对话资料HY评分之准确度

import os
import json
import random
import csv
from typing import List, Dict
from tqdm import tqdm

from openai import OpenAI
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDER_A = os.environ.get(
    "GEN_TURNS_DIR",
    os.path.join(BASE_DIR, "HY_single_turn_access", "Purely_Gen_single_turn_jsons_copy"),
)
# Reference real turns are private and should not be committed.
FOLDER_B = os.environ.get(
    "REF_TURNS_DIR",
    os.path.join(BASE_DIR, "private_reference_turns"),
)
OUTPUT_FILE = "evaluation_result.csv"

OUTPUT_TRUE_FOLDER = "./GPT_assessment_True_1"
OUTPUT_FALSE_FOLDER = "./GPT_assessment_False_1"

os.makedirs(OUTPUT_TRUE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FALSE_FOLDER, exist_ok=True)


def load_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_turn_refs(ref_path: str, num_refs: int = 2) -> List[Dict]:
    try:
        turns = load_json(ref_path)
        return random.sample(turns, min(len(turns), num_refs))
    except Exception:
        return []


def build_prompt(target_turn: Dict, ref_turns: List[Dict]) -> str:
    prompt = """
        你是一个心理学专家，我会给你一段护理师和病患的单轮对话，以及标注分数，请根据下面的参考对话判断对病患(P)回应的标注分数是否合理。

        【当前对话】
        护理师(N): {N}
        病患(P): {P}
        标注分数: {score}

        【参考对话】
        """.format(N=target_turn["N"], P=target_turn["P"], score=target_turn["score"])

    for i, ref in enumerate(ref_turns):
        prompt += f"\n参考{i+1} - 护理师(N): {ref['N']}\n病患(P): {ref['P']}"

    prompt += """
            注意：【参考对话】结构与【当前对话】类似，并且【参考对话】中病患P的回应，均符合目前的标注分数标准:{score}
            请根据【参考对话】内容，判断当前对话中，对病患(P)的标注分数是否准确，若明显错误，请回复 "0"，否则回复 "1"。
            """
    return prompt.strip()


def ask_gpt(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的心理对话评估专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def evaluate_all():
    rows = ["project_id,file_name,turn_id,score,gpt_judgment"]

    for project_id in os.listdir(FOLDER_A):
        subdir_a = os.path.join(FOLDER_A, project_id)
        subdir_b = os.path.join(FOLDER_B, project_id)
        if not os.path.isdir(subdir_a):
            continue

        for file in os.listdir(subdir_a):
            if not file.endswith(".json"):
                continue

            print(f"处理文件: {project_id}/{file}")

            file_a = os.path.join(subdir_a, file)
            file_b = os.path.join(subdir_b, file)
            data_a = load_json(file_a)

            true_turns = []
            false_turns = []

            if not os.path.exists(file_b):
                for turn in tqdm(data_a, desc=f"{project_id}/{file}"):
                    turn["gpt_judgment"] = 1
                    true_turns.append(turn)
            else:
                for turn_id, turn in enumerate(tqdm(data_a, desc=f"{project_id}/{file}")):
                    refs = get_turn_refs(file_b, num_refs=3)
                    if not refs:
                        turn["gpt_judgment"] = 1
                        true_turns.append(turn)
                        continue

                    prompt = build_prompt(turn, refs)
                    judgment = ask_gpt(prompt)
                    if judgment not in ["0", "1"]:
                        judgment = "1"
                    turn["gpt_judgment"] = int(judgment)
                    rows.append(f"{project_id},{file},{turn_id},{turn['score']},{judgment}")

                    if turn["gpt_judgment"] == 1:
                        true_turns.append(turn)
                    else:
                        false_turns.append(turn)

            if true_turns:
                output_true_path = os.path.join(OUTPUT_TRUE_FOLDER, project_id)
                os.makedirs(output_true_path, exist_ok=True)
                save_json(os.path.join(output_true_path, file), true_turns)

            if false_turns:
                output_false_path = os.path.join(OUTPUT_FALSE_FOLDER, project_id)
                os.makedirs(output_false_path, exist_ok=True)
                save_json(os.path.join(output_false_path, file), false_turns)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


if __name__ == "__main__":
    evaluate_all()
    print(f"评估完成，结果保存在 {OUTPUT_FILE}")


