# -*- coding: utf-8 -*-
import sys 
sys.path.append('../memory_bank')
# from azure_client import LLMClientSimple
import openai, json, os
import argparse
import copy
from typing import List, Dict

from MEMORY_BANK.model_loader import get_model , generate_respond

class LLMClientSimple:

    def __init__(self,gen_config=None):
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.disable_tqdm = False
        self.gen_config = gen_config 

    def generate_text_simple(self,prompt,prompt_num,language='en'):
        self.gen_config['n'] = prompt_num
        retry_times,count = 5,0
        response = None
        while response is None and count<retry_times:
            try:
                request = copy.deepcopy(self.gen_config)
                # print(prompt)
                if language=='cn':
                    message = [
                    {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
                    {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
                    {"role": "system", "content": "好的，我会尽力帮你的。"},
                    {"role": "user", "content": f"{prompt}"}]
                else:
                    message = [
                    {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                    {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                    {"role": "system", "content": "Sure, I will do my best to assist you."},
                    {"role": "user", "content": f"{prompt}"}]
                response = openai.ChatCompletion.create(
                    **request, messages=message)
                # print(prompt)
            except Exception as e:
                print(e)
                if 'This model\'s maximum context' in str(e):
                        cut_length = 1800-200*(count)
                        print('max context length reached, cut to {}'.format(cut_length))
                        prompt = prompt[-cut_length:]
                        response=None
                count+=1
        if response:
            task_desc = response['choices'][0]['message']['content'] #[response['choices'][i]['text'] for i in range(len(response['choices']))]
        else:
            task_desc = ''
        return task_desc
    

chatgpt_config = {"model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 400,
        "top_p": 1.0,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        "stop": ["<|im_end|>", "¬人类¬"]
        }

llm_client = LLMClientSimple(chatgpt_config)

def load_prompt_from_file(file_path,user_name):
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()
        content = content.replace("{user_name}", user_name)
        #content = f.read().strip()
        return [{"role": "system", "content": content}]

def summarize_content_prompt(content,user_name,boot_name,language='cn'):
    prompt : List[Dict[str, str]] = []
    prompt = load_prompt_from_file('summarization.txt',user_name)
    prompt1 = ''
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt1 += f"\n{user_name}：{query.strip()}"
        prompt1 += f"\n{boot_name}：{response.strip()}"
    prompt1 += ('\n总结：')
    prompt.append({"role": "user", "content": prompt1})
    return prompt

def summarize_overall_prompt(content):
    prompt_so : List[Dict[str, str]] = []
    prompt_so = [{"role": "system", "content": '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n'}]
    prompt_so1=''
    for date,summary_dict in content:
        summary = summary_dict['content']
        prompt_so1 += (f"\n时间{date}发生的事件为{summary.strip()}")
    prompt_so1 += ('\n只输出事件本身描述，总结：')
    prompt_so.append({"role": "user", "content": prompt_so1})
    return prompt_so

def summarize_overall_personality(content):
    prompt_sop : List[Dict[str, str]] = []
    prompt_sop = [{"role": "system", "content":  '以下是用户在多段对话中展现出来的个性：\n'}]
    prompt_sop1=''
    for date,summary in content:
        prompt_sop1 += (f"\n在时间{date}的分析为{summary['content'].strip()}")
    prompt_sop1 += ('\n请总体概括用户的个性，尽量简洁精炼，高度概括。总结为：')
    prompt_sop.append({"role": "user", "content": prompt_sop1})
    return prompt_sop

def summarize_person_prompt(content,user_name,boot_name,language):
    prompt_p : List[Dict[str, str]] = []
    prompt_p = load_prompt_from_file('personality.txt',user_name)
    prompt_p_1=''
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt_p_1 += f"\n{user_name}：{query.strip()}"
        prompt_p_1 += f"\n{boot_name}：{response.strip()}"

    prompt_p_1 += (f'\n{user_name}的个性：')
    prompt_p.append({"role": "user", "content": prompt_p_1})
    return prompt_p


def summarize_memory(model, tokenizer,memory_dir,name=None,language='cn'):
    boot_name = '心理咨询师'
    gen_prompt_num = 1
    memory = json.loads(open(memory_dir,'r',encoding='utf8').read())
    all_prompts,all_his_prompts, all_person_prompts = [],[],[]
    gen_kwargs = {"temperature": 0.5,
                    "max_new_tokens": 64,
                    "top_p": 0.7,
                    "top_k": 20,
                    "length_penalty": 0.3,
                    "do_sample": True,
                    "num_beams": 3,
                    "no_repeat_ngram_size": 2
                }
    for k,v in memory.items():
        if name != None and k != name:
            continue
        user_name = k
        print(f'Updating memory for user {user_name}')
        if v.get('history') == None:
            continue
        history = v['history']
        if v.get('summary') == None:
            memory[user_name]['summary'] = {}
        if v.get('personality') == None:
            memory[user_name]['personality'] = {}

        for date, content in history.items():
            # print(f'Updating memory for date {date}')
            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True
            hisprompt = summarize_content_prompt(content,user_name,boot_name,language)
            person_prompt = summarize_person_prompt(content,user_name,boot_name,language)
            if his_flag:
                #his_summary = llm_client.generate_text_simple(prompt=hisprompt,prompt_num=gen_prompt_num,language=language)
                his_summary=generate_respond(tokenizer, model,hisprompt,gen_kwargs)
                print(his_summary)
                memory[user_name]['summary'][date] = {'content':his_summary}
            if person_flag:
                #person_summary = llm_client.generate_text_simple(prompt=person_prompt,prompt_num=gen_prompt_num,language=language)
                person_summary=generate_respond(tokenizer, model,person_prompt,gen_kwargs)
                print(person_summary)
                memory[user_name]['personality'][date] = {'content':person_summary}
        
        overall_his_prompt = summarize_overall_prompt(list(memory[user_name]['summary'].items()))
        overall_person_prompt = summarize_overall_personality(list(memory[user_name]['personality'].items()))
        memory[user_name]['overall_history'] = generate_respond(tokenizer, model, overall_his_prompt, gen_kwargs)
        print(memory[user_name]['overall_history'])
        memory[user_name]['overall_personality'] = generate_respond(tokenizer, model, overall_person_prompt, gen_kwargs)
        print(memory[user_name]['overall_personality'])
 
    with open(memory_dir,'w',encoding='utf8') as f:
        print(f'Sucessfully update memory for {name}')
        json.dump(memory,f,ensure_ascii=False, indent=4)
    return memory

if __name__ == '__main__':
    summarize_memory('../memories/eng_memory_cases.json',language='en')


                


