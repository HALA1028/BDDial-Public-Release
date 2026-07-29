import os
import re
import outlines
from openai import OpenAI
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 更新后的提取标签正则表达式
TAG_PATTERN = r"[【\[]?(H|Y)(\d+)(?:[.\s]+[^】\]]+)?[】\]]?"


# 支持 N: 与 N： 以及 P: 与 P：的正则表达式
QUESTION_PREFIX = r"^(N[:：])"
ANSWER_PREFIX = r"^(P[:：])"

# 加载评分规则
def load_rules(rules_folder):
    rules = {}
    for file_name in os.listdir(rules_folder):
        if file_name.endswith('.txt'):
            tag_id = file_name.split('.')[0]  # 提取编号，例如 H23
            file_path = os.path.join(rules_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                rules[tag_id] = file.read().strip()  # 读取评分规则
    return rules

# 调用 GPT-4 API
def query_gpt4(n_question, p_answer, rule_text, tag_id):
    prompt = f"""你是一位心理学专家，请根据评分规则、对话问题、对话答案进行对话评分的选择。规则如下：

                评分规则：
                {rule_text}

                对话问题：
                {n_question}

                对话回答：
                {p_answer}

                请你先思考选择的理由，但禁止输出理由，最后选择对于回答的评分：
    
                """

    try: 
        model = outlines.models.openai(
            "gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        if tag_id in {"H1","H4", "H5", "H6", "H12", "H14", "H13", "H16", "H17", "H18", "H21"}:
            generator = outlines.generate.choice(model, ["0", "1", "2"])
        elif tag_id in {"Y5", "Y6", "Y8"}:
            generator = outlines.generate.choice(model, ["0", "2", "4", "6", "8"])
        elif tag_id in {"Y9"}:
            generator = outlines.generate.choice(model, ["0", "2", "4", "6"])
        elif tag_id in {"H10", "Y1", "Y3"}:
            generator = outlines.generate.choice(model, ["0", "1", "2", "3"])
        elif tag_id in {"Y2"}:
            generator = outlines.generate.choice(model, ["0", "1"])
        else:
            generator = outlines.generate.choice(model, ["0", "1", "2", "3", "4"])
        answer = generator(prompt)

        return answer
    except Exception as e:
        print(f"调用 GPT-4 API 出现错误：{e}")
        return None

# 处理单个文件并更新内容
def process_file(file_path, rules, output_folder):
    new_content = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 临时存储当前的 N 和 P 对话
    current_question = None
    current_answer = None
    
    for i in range(len(lines)):
        line = lines[i].strip()

        # 处理问题行，支持 N: 与 N：
        if re.match(QUESTION_PREFIX, line):
            current_question = re.sub(QUESTION_PREFIX, "", line).strip()
            tags = re.findall(TAG_PATTERN, current_question)  # 匹配所有标签
            if tags:
                new_content.append(line + '\n')  # 保留问题行

                # 查找回答行
                if i + 1 < len(lines):
                    current_answer = None
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if re.match(ANSWER_PREFIX, next_line):
                            current_answer = re.sub(ANSWER_PREFIX, "", next_line).strip()
                            break
                        elif re.match(QUESTION_PREFIX, next_line):
                            break
                        j += 1

                    if current_answer:
                        scores = []
                        for tag in tags:
                            tag_type = tag[0]  # H 或 Y
                            tag_number = tag[1]  # 编号数字
                            tag_id = f"{tag_type}{tag_number}"
                            rule_text = rules.get(tag_id, "暂无评分规则")

                            gpt_output = query_gpt4(current_question, current_answer, rule_text, tag_id)
                            print(gpt_output + '\n')

                            if gpt_output:
                                try:
                                    score = int(gpt_output.strip())
                                    scores.append(f"<{tag_id}={score}>")
                                except ValueError:
                                    scores.append(f"<{tag_id}=0>")
                            else:
                                scores.append(f"<{tag_id}=0>")

                        new_content.append(f"P: {current_answer}{' '.join(scores)}\n")
                    else:
                        new_content.append(lines[i] + "\n")
            else:
                new_content.append(line + "\n")
        elif line.strip() == "":
            continue
        else:
            if re.match(ANSWER_PREFIX, line):
                last_content = new_content[-1] if len(new_content) > 0 else ""
                if re.match(r"P: .+<.+=\d+>", last_content):
                    continue

            new_content.append(line + "\n")

    output_file = os.path.join(output_folder, os.path.basename(file_path))
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(new_content)
    print(f"文件已更新：{output_file}")

# 遍历文件夹并处理所有文件
def process_folder(txt_folder, rules_folder, output_folder):
    rules = load_rules(rules_folder)  # 加载评分规则
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹
    for file_name in os.listdir(txt_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(txt_folder, file_name)
            process_file(file_path, rules, output_folder)

# 示例调用
txt_folder = os.path.join(BASE_DIR, "stage2_formal_output_v2_selected")
rules_folder = os.path.join(BASE_DIR, "HY_rules")
output_folder = os.path.join(BASE_DIR, "stage3_formal_output_v2_selected")
process_folder(txt_folder, rules_folder, output_folder)
print(f"所有文件已处理，结果保存至 {output_folder}")
