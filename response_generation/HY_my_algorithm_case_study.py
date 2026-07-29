import datetime
import os
import re
import shutil
import logging
import sys
import time, platform
import signal,json
import gradio as gr
import nltk
from transformers.generation.utils import LogitsProcessorList
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import jieba
from bert_score import score as bert_score
from nltk import ngrams
from collections import defaultdict
import nltk
import numpy as np
nltk.download('punkt')
from sentence_transformers import SentenceTransformer

prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
print(bank_path)
sys.path.append(prompt_path)
sys.path.append(bank_path)
from MEMORY_BANK.utils.prompt_utils import *
from MEMORY_BANK.utils.memory_utils import enter_name, summarize_memory_event_personality, save_local_memory
from MEMORY_BANK.utils.model_utils import InvalidScoreLogitsProcessor
from MEMORY_BANK.utils.sys_args import data_args,model_args
from MEMORY_BANK.memory_retrieval.local_doc_qa import reset_selected_questions
from MEMORY_BANK.memory_retrieval.local_doc_qa_HY_select_algorithm import adaptive_hy_selection

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

current_path = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_path, ".."))
hy_scoring_path = os.path.join(repo_root, "HY_scoring")
if hy_scoring_path not in sys.path:
    sys.path.append(hy_scoring_path)

from MEMORY_BANK.memory_bank.memory_retrieval.configs.model_config import *
from MEMORY_BANK.model_loader import get_model, generate_respond

from HY_Scoring.HY_Scoring import InferenceEngine #HY评分模型

# 初始化推理引擎
engine = InferenceEngine(
    model_path=os.environ.get("HY_SCORING_MODEL_PATH", os.path.join(repo_root, "HY_scoring", "HY_Scoring", "fold5")),
    item_label_map_file=os.path.join(repo_root, "HY_scoring", "HY_Scoring", "label_mappings.json"),
    output_file="result.csv"
)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
if not os.path.exists(memory_dir):
    json.dump({},open(memory_dir,"w",encoding="utf-8"))

language = data_args.language
if data_args.enable_forget_mechanism:
    from MEMORY_BANK.memory_retrieval.forget_memory import LocalMemoryRetrieval
else:
    from MEMORY_BANK.memory_retrieval.local_doc_qa_HY_select_algorithm import LocalMemoryRetrieval

local_memory_qa = LocalMemoryRetrieval()#用于存储和检索用户记忆
EMBEDDING_MODEL = EMBEDDING_MODEL_CN if language == 'cn' else EMBEDDING_MODEL_EN
local_memory_qa.init_cfg(
                        embedding_model=EMBEDDING_MODEL,
                        embedding_device=EMBEDDING_DEVICE,
                        top_k=data_args.memory_search_top_k,#最多检索出 K 条最相关的记忆
                        language=language)

meta_prompt = generate_meta_prompt_dict_chatglm_app()[language]#老用户
personality_prompt= generate_personality_prompt()[language]
HY_prompt = generate_HY_prompt()[language]
HY_Memory_prompt = generate_HY_Memory_prompt()[language]
realate_peompt = generate_realate_peompt()[language]
relate_Memory_peompt = generate_relate_Memory_peompt()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatglm()#新用户
user_keyword = generate_user_keyword()[language] #用户名字
ai_keyword = generate_ai_keyword()[language] #机器人名字
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_first_bracket_content(text):
    match = re.search(r'\[(.*?)\]', text)
    return match.group(1) if match else None


tokenizer, model = get_model()

def build_prompt(text,user_memory,user_name,user_memory_index,
                local_memory_qa,
                meta_prompt,personality_prompt,
                HY_prompt,
                HY_Memory_prompt,
                relate_peompt,
                relate_Memory_peompt,
                new_user_meta_prompt,
                user_keyword,
                ai_keyword,
                boot_actual_name,
                selected_questions,I,i_rel,t,count
                    ):
    judge=0
    Memory_item = False
    memory_search_query = text #f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    #HY_item=''

    '''
    if user_HY_index:
        related_HY,aa,HY_item,matched_item = local_memory_qa.search_HY(memory_search_query,user_HY_index,selected_questions)
        related_HY = '\n'.join(related_HY)
    else:
        related_HY = ""
    '''
    HY_item, matched_item, related_HY = adaptive_hy_selection(text, I, t, count) #寻找合适的HY问题
    if related_HY != '':
        related_HY_content = f"\n{str(related_HY).strip()}\n"
    else:
        related_HY_content = ''

    #寻找记忆#####################################################################################################
    if user_memory_index:
        related_memos, memo_dates,Memory_item= local_memory_qa.search_memory(memory_search_query,user_memory_index)
        related_memos = '\n'.join(related_memos)
    else:
        related_memos = ""
    ##############################################################################################################
    if "overall_history" in user_memory:
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) 
    else:
        history_summary = ''

    if related_memos != '':
        related_memory_content = f"\n{str(related_memos).strip()}\n"
    else:
        related_memory_content = ''
        
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""

    if related_HY_content:
        judge=1
        if related_memory_content:
            prompt1 = HY_Memory_prompt.format(text=text,related_HY_content=related_HY_content,matched_item=matched_item,related_memory_content=related_memory_content)
            new_user_meta_prompt.append({"role": "system", "content": prompt1})
            prompt = new_user_meta_prompt
        else:
            prompt1 = HY_prompt.format(text=text,related_HY_content=related_HY_content,matched_item=matched_item)
            new_user_meta_prompt.append({"role": "system", "content": prompt1})
            prompt = new_user_meta_prompt

    else:
        if i_rel and related_memory_content:
            judge=1
            matched_item = I[i_rel]["Item"]
            related_HY_content = I[i_rel]["Description"]
            prompt1 = relate_Memory_peompt.format(text=text,related_HY_content=related_HY_content,matched_item=matched_item,related_memory_content=related_memory_content)
            new_user_meta_prompt.append({"role": "system", "content": prompt1})
            prompt = new_user_meta_prompt

        elif i_rel:
            judge=1
            matched_item = I[i_rel]["Item"]
            related_HY_content = I[i_rel]["Description"]
            prompt1 = relate_peompt.format(text=text,related_HY_content=related_HY_content,matched_item=matched_item)
            new_user_meta_prompt.append({"role": "system", "content": prompt1})
            prompt = new_user_meta_prompt

        elif history_summary and related_memory_content and personality:
                judge=1
                prompt1 = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=text,memo_dates=memo_dates)
                new_user_meta_prompt.append({"role": "system", "content": prompt1})
                prompt = new_user_meta_prompt
        else:
            prompt = new_user_meta_prompt

    return prompt,judge,HY_item,Memory_item

def chat(model, tokenizer, query: str, new_user_meta_prompt, 
         I,i_rel,t, count,
         user_memory=None,
         user_name=None,
         user_memory_index=None, 
         selected_questions=None,
         logits_processor=None
         ):
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {
            "num_beams": 3, "do_sample": True, "top_p": 0.7,
            "temperature": 0.7, "top_k": 20, "length_penalty": 0.3,
            "no_repeat_ngram_size": 3, "num_return_sequences": 1,
            "repetition_penalty": 1.5, "max_new_tokens": 64
        }

    prompt,judge,HY_item,Memory_item = build_prompt(query,user_memory,user_name,user_memory_index,                                                       
                                            local_memory_qa,
                                            meta_prompt,personality_prompt,
                                            HY_prompt,
                                            HY_Memory_prompt,
                                            realate_peompt,
                                            relate_Memory_peompt,
                                            new_user_meta_prompt,
                                            user_keyword,
                                            ai_keyword,
                                            boot_actual_name,
                                            selected_questions, I, i_rel, t, count
                                            )
    start_time = time.time()
    response = generate_respond(tokenizer,model,prompt,gen_kwargs)
    end_time = time.time()

    generation_time = end_time - start_time

    if judge == 1:
        prompt.pop()

    response = clean_result(response,prompt,stop_words=[user_keyword])
    return response,HY_item,Memory_item, generation_time
    
def clean_result(result,prompt,stop_words):

    result = result.replace("&nbsp;","")
    result = result.replace("\n","")
    for stop in stop_words:
        if stop in result:
            result = result[:result.index(stop)].strip()
    result = result.replace(ai_keyword,"").strip()
    result = result.replace(":","").strip()
    result = result.replace("：","").strip()

    # 保留中文（含繁体）、英文、空格、中英文标点符号
    result = re.sub(r"[^\u4e00-\u9fffA-Za-z\s"
                    r"\u3000-\u303F"  # 中文符号
                    r"\uFF00-\uFFEF"  # 全角英数符号（包括中文标点）
                    r"\u2000-\u206F"  # 常用标点
                    r"\u0020-\u007F"  # 基本拉丁文（包括英文标点）
                    r"]", "", result)
    # print(result)
    return result

def HY(name, memory,local_memory_qa,data_args,update_memory_index=True):
    cur_date = datetime.date.today().strftime("%Y-%m-%d")
    user_memory_index = None
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    if isinstance(local_memory_qa,gr.State):
        local_memory_qa=local_memory_qa.value
    memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)

    if name in memory.keys():
        user_memory = memory[name]
        memory_index_path = os.path.join(data_args.memory_basic_dir,f'memory_index/{name}_index')
        os.makedirs(os.path.dirname(memory_index_path), exist_ok=True)
        if (not os.path.exists(memory_index_path)) or update_memory_index:
            print(f'Initializing memory index {memory_index_path}...')
            if os.path.exists(memory_index_path):
                shutil.rmtree(memory_index_path)
            memory_index_path, _ = local_memory_qa.init_memory_vector_store(filepath=memory_dir,vs_path=memory_index_path,user_name=name,cur_date=cur_date)                      
        
        user_memory_index = local_memory_qa.load_memory_index(memory_index_path) if memory_index_path else None
        msg = f"欢迎回来，{name}！" if data_args.language=='cn' else f"Wellcome Back, {name}！"
        return msg,user_memory, memory, name , user_memory_index
    else:
        memory[name] = {}
        memory[name].update({"name":name}) 
        msg = f"欢迎新用户{name}！我会记住你的名字，下次见面就能叫你的名字啦！" if data_args.language == 'cn' else f'Welcome, new user {name}! I will remember your name, so next time we meet, I\'ll be able to call you by your name!'
        return msg,memory[name],memory,name,user_memory_index

def parse_filename(filename):
    match = re.match(r"\d{6}_(\d+)\.txt", filename)
    if match:
        return match.group(1)  # 提取病人 ID 作为 user_name
    return None

def load_user_memory_directly(memory_dir, user_name):
    """
    实验：直接从 memory_dir 文件中加载 user_name 对应的 memory 数据
    """
    # 检查文件是否存在
    if not os.path.exists(memory_dir):
        raise FileNotFoundError(f"Memory file not found: {memory_dir}")

    # 读取 JSON 文件
    with open(memory_dir, 'r', encoding='utf8') as f:
        memory = json.load(f)

    # 提取 user_name 对应的 memory 数据
    user_memory = memory.get(user_name, {})
    return user_memory

sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def preprocess_I(I):
    """ 预计算 I 中的嵌入并缓存 """
    for key, value in I.items():
        # 计算 Description 的嵌入
        value["Description_emb"] = sbert_model.encode(value['Description'])

        # 计算 Key_word 的嵌入并存储到 Key_word_emb
        key_words = value["Key_word"]
        if isinstance(key_words, list):  
            value["Key_word_emb"] = sbert_model.encode(key_words)
        elif isinstance(key_words, str):  
            value["Key_word_emb"] = sbert_model.encode([key_words])
        else:
            value["Key_word_emb"] = None

    return I

def main():
    # 初始化结构
    global new_user_meta_prompt
    global memory
    global selected_questions

     # 读取 HY 信息
    with open(os.path.join(current_path, 'memories', 'updated_file2.json'), 'r', encoding='utf-8') as f:
        I = json.load(f)
    I = preprocess_I(I)
    with open(os.path.join(current_path, 'evaluation', 'Relevance_Matrix.json'), 'r', encoding='utf-8') as f:
        Relate = json.load(f)

    memory = json.loads(open(memory_dir, "r", encoding="utf-8").read())

    user_name = input("请输入用户名称（user_name）：").strip()
    if user_name in memory.keys():
        user_memory = load_user_memory_directly(memory_dir, user_name)
    msg, user_memory, memory, user_name, user_memory_index = HY(user_name, memory, local_memory_qa, data_args)

    new_user_meta_prompt = []
    selected_questions = set()
    count = {} # 记录问题被问过的次数
    t = 0
    i_rel = ''
    HY_item = ''
   
    print(msg)
    print("对话系统初始化完毕。请输入你的问题（输入 q 或 quit 退出）：")

    while True:

        user_input = input("\n你：").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("已退出对话系统。")
            summarize_memory_event_personality(model,tokenizer,data_args, memory, user_name)
            break
        
        if HY_item:  
            print("进入评分流程：")
            HY_score = engine.infer(HY_item, generated_n, user_input)
            print (f"HY评分结果：{HY_score}")
            HY_item = ""   



        new_user_meta_prompt.append({"role": "user", "content": user_input})

        generated_n, HY_item, Memory_item, generation_time = chat(
            model, tokenizer, user_input, new_user_meta_prompt,
            I, i_rel, t, count,
            user_name=user_name,
            user_memory=user_memory,
            user_memory_index=user_memory_index,
            selected_questions=selected_questions
        )

        print("N：", generated_n)
        i_rel=''

        new_user_meta_prompt.append({"role": "assistant", "content": generated_n})

        # 处理 i_rel 更新
        if HY_item:
            #selected_questions.update(dialogue["HY"])
            count[HY_item] = count.get(HY_item, 0) + 1
            related_scores = {}
            if HY_item in Relate:
                for rel in Relate[HY_item]:
                    j = rel["item"]
                    related_scores[j] = rel["relevance"] * (0.5 ** count.get(j, 0))

            # 找到分数最高的相关问题（如果存在）
            most_related = max(related_scores, key=related_scores.get) if related_scores else None
            # 设定 0.5 分数门槛
            i_rel = most_related if (most_related and related_scores[most_related] >= 0.5) else ''

        t += 1#记录轮次

        # 控制历史轮数
        if len(new_user_meta_prompt) > 10:
            new_user_meta_prompt = [new_user_meta_prompt[0]] + new_user_meta_prompt[-9:]

if __name__ == "__main__":
    main()
