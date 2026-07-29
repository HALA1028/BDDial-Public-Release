import os
from typing import List, Dict

RESPONSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def prompt_file(name):
    return os.path.join(RESPONSE_ROOT, "system_prompt", name)

def response_file(name):
    return os.path.join(RESPONSE_ROOT, name)

boot_name_dict = {'en':'AI Companion','cn':'AI伴侣'}
boot_actual_name_dict = {'en':'BDDial Assistant','cn':'BDDial Assistant'}
def output_prompt(history,user_name,boot_name):
    prompt = f"我是你的AI伴侣{boot_name}，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n\n{user_name}：{query}"
        prompt += f"\n\n{boot_name}：{response}"
    return prompt
    
def generate_meta_prompt_dict_chatglm_app(**kwargs):
    # meta_prompt_dict = {'cn':"""
    # 本轮你将结合用户{user_name}的回忆进行回答，你应该做到：
    #                     （1）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。
    #                     （2）输出必须保持精简，以推动聊天顺利进行
    #                     （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。
    # \n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：
    #                     “{related_memory_content}\n记忆中这段[回忆]的日期为{memo_dates}。”
    #                     以下是你（{boot_actual_name}）与用户{user_name}的多轮对话。你应该参考对话上下文，过去的[回忆]，详细回复用户问题。
    # 回复的表达方式请参考用户{user_name}的个性：{personality}
    # """}  
    with open(prompt_file("memory_prompt.txt"), "r", encoding="utf-8") as file:
        prompt_template = file.read()
    meta_prompt_dict = {'cn': prompt_template}
    return meta_prompt_dict

def generate_personality_prompt():
    personality_prompt_dict = {'cn':"""
    本轮你将结合用户{user_name}的个性进行回答，你应该做到：
                        （1）你必须根据用户的个性，思考回复用户的策略。
                        （2）根据这个策略，对用户进行回复，你只能输出回复本身，禁止输出策略
                        （3）输出必须保持精简，以推动聊天顺利进行
    \n回复的表达方式请参考用户{user_name}的个性：{personality}
    """}  

    return personality_prompt_dict

def generate_HY_prompt():
    # HY_prompt_dict = {'cn':"""
    # 本轮你将根据{related_HY_content}的内部描述对用户进行提问，你应该做到：
    #                     （1）你必须结合[related_HY_content]状况描述，思考如何向用户进行提问。
    #                     （2）你必须询问[related_HY_content]的状况具体的程度
    #                     （3）你的提问必须保持精简，以推动评测的顺利进行
    # """}  
    with open(prompt_file("HY_prompt.txt"), "r", encoding="utf-8") as file:
        prompt_template = file.read()
    HY_prompt_dict = {'cn': prompt_template}
    return HY_prompt_dict

def generate_HY_Memory_prompt():
    with open(prompt_file("HY_Memory_prompt.txt"), "r", encoding="utf-8") as file:
        prompt_template = file.read()
    HY_Memory_prompt_dict = {'cn': prompt_template}
    return HY_Memory_prompt_dict

def generate_realate_peompt():
    with open(prompt_file("relate_prompt.txt"), "r", encoding="utf-8") as file:
        prompt_template = file.read()
    realate_prompt_dict = {'cn': prompt_template}
    return realate_prompt_dict

def generate_relate_Memory_peompt():
    with open(prompt_file("relate_Memory_prompt.txt"), "r", encoding="utf-8") as file:
        prompt_template = file.read()
    realate_Memory_prompt_dict = {'cn': prompt_template}
    return realate_Memory_prompt_dict


def generate_meta_prompt_dict_chatglm_belle_eval():
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。\
    （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\
    用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：\
    “{related_memory_content}\n记忆中这段[回忆]的日期为{memo_dates}。”以下是你（{boot_actual_name}）与用户{user_name}的多轮对话。\
    人类的问题以[|用户|]: 开头，而你的回答以[|AI伴侣|]开头。你应该参考对话上下文，过去的[回忆]，详细回复用户问题，以下是一个示例：\
    1.（用户提问）[|用户|]: 你还记得我5月4号看了什么电影？\n2.据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是:\
    “[|AI伴侣|]：你喜欢看电影吗？\n[|用户|]：我喜欢看电影，我今天去看了《猩球崛起》，特别好看。”\n记忆中这段[回忆]的日期为5月4日\n”\
    3.(你的回答) [|AI伴侣|]：你在5月4日去看了《猩球崛起》，特别好看。\
    请你参考示例理解并使用[回忆]，以如下形式开展对话： [|用户|]: 你好! \
    [|AI伴侣|]: 你好呀，我的名字是{boot_actual_name}! {history_text}
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\nThe date of this [memory] in the memory is {memo_dates}." Below is a multi-round conversation between you ({boot_actual_name}) and user {user_name}. You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. Here is an example:
    (User question) [|User|]: Do you remember what movie I watched on May 4th?\n2. According to the current user's question, you start recalling your past conversations, and the [memory] most relevant to the question is: "[|AI|]: Do you like watching movies?\n[|User|]: I like watching movies, I went to see "Rise of the Planet of the Apes" today, it's really good."\nThe date of this [memory] in the memory is May 4th\n"3. (Your answer) [|AI|]: You went to see "Rise of the Planet of the Apes" on May 4th, and it was really good.
    Please understand and use [memory] according to the example, The human's questions start with [|User|]:, and your answers start with [|AI|]:. Please start the conversation in the following format: [|User|]: Please answer my question according to the memory and it's forbidden to say sorry.\n[|AI|]: Sure!\n {history_text}
    """} 
    return meta_prompt_dict

def generate_meta_prompt_dict_chatgpt():
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。\
    （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\
    用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：
    “{related_memory_content}\n"。
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\n"  You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. 
    """} 
    return meta_prompt_dict

def generate_new_user_meta_prompt_dict_chatgpt():
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；\
    （2）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。"。
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    """} 
    return meta_prompt_dict

# def generate_meta_prompt_dict_chatgpt_cli():
#     meta_prompt_dict =  {'cn':"""
#     现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。（3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。
#     用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：“{related_memory_content}\n"。
#     """,
#     'en':"""
#     Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
#     The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\n"  You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. 
#     """} 
#     return meta_prompt_dict

def generate_user_keyword():
    return {'cn': '[|用户|]', 'en': '[|User|]'}

def generate_ai_keyword():
    return {'cn': '[|AI伴侣|]', 'en': '[|AI|]'}

def generate_new_user_meta_prompt_dict_chatglm():

    chat_history: List[Dict[str, str]] = []
    
    chat_history=load_prompt_from_file(response_file('tests_system_memory_best.txt'))

    return chat_history

def load_prompt_from_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()
        #content = f.read().strip()
        return [{"role": "system", "content": content}]


def build_prompt_with_search_memory_chatglm_app(history,
                                                text,user_memory,user_name,user_memory_index,user_HY_index,
                                                local_memory_qa,
                                                meta_prompt,personality_prompt,HY_prompt,
                                                new_user_meta_prompt,
                                                user_keyword,
                                                ai_keyword,
                                                boot_actual_name,
                                                language):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text#f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    if user_memory_index:
        related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
        related_memos = '\n'.join(related_memos)
    else:
        related_memos = ""

    if user_HY_index:
        related_HY, memo_dates= local_memory_qa.search_HY(memory_search_query,user_HY_index)
        related_HY = '\n'.join(related_HY)
    else:
        related_HY = ""
  
 
    if "overall_history" in user_memory:
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) 
    #if "history" in user_memory:
    #    history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["history"]) 
    else:
        history_summary = ''
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    if related_memos != '':
        related_memory_content = f"\n{str(related_memos).strip()}\n"
    else:
        related_memory_content = ''

    if related_HY != '':
        related_HY_content = f"\n{str(related_HY).strip()}\n"
    else:
        related_HY_content = ''

    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
    #personality = user_memory['personality'] if "personality" in user_memory else ""
   
    history_text = ''
    for dialog in history:
        query = dialog[0]
        response = dialog[1]
        new_user_meta_prompt.append({"role": "user", "content": query})
        new_user_meta_prompt.append({"role": "assistant", "content": response})
    new_user_meta_prompt.append({"role": "user", "content": text})

    if related_HY_content:
        prompt1 = HY_prompt.format(user_name=user_name,history_text=text,related_HY_content=related_HY_content)
        new_user_meta_prompt.append({"role": "system", "content": prompt1})
        prompt = new_user_meta_prompt
    else:
        if history_summary and related_memory_content and personality:
            prompt1 = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=text,memo_dates=memo_dates)
            new_user_meta_prompt.append({"role": "system", "content": prompt1})
            prompt = new_user_meta_prompt
        elif personality:
            prompt1 = personality_prompt.format(user_name=user_name,personality=personality,history_text=text)
            new_user_meta_prompt.append({"role": "system", "content": prompt1})
            prompt = new_user_meta_prompt
        else:
            prompt = new_user_meta_prompt
            #for message in prompt:
                #if "content" in message:
                    #message["content"] = message["content"].format(user_name=user_name)
        # print(prompt)
    return prompt

def build_prompt_with_search_memory_chatglm_eval(history,text,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,user_keyword,ai_keyword,boot_actual_name,language):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text#f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
    related_memos = '\n'.join(related_memos)
    related_memos = related_memos.replace('Memory:','').strip()  
    
    history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) \
        if language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    related_memory_content = f"\n{str(related_memos).strip()}\n"
    personality = user_memory['overall_personality']
    history_text = ''
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        history_text += f"\n {user_keyword}: {query}"
        history_text += f"\n {ai_keyword}: {response}"
    history_text += f"\n {user_keyword}: {text} \n {ai_keyword}: " 
    prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=history_text,memo_dates=memo_dates)
    # print(prompt)
    return prompt,related_memos


def build_prompt_with_search_memory_belle_eval(history,text,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,new_user_meta_prompt,user_keyword,ai_keyword,boot_actual_name,language):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text#f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
    related_memos = '\n'.join(related_memos)
    # print(f'\n{text}\n----------\n',related_memos,'\n----------\n')
    # response = user_memory_index.query(memory_search_query,service_context=service_context)
    # print(response)
 
    history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if language=='cn' \
     else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    related_memory_content = f"\n{str(related_memos).strip()}\n"
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
    
    history_text = ''
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        history_text += f"\n {user_keyword}: {query}"
        history_text += f"\n {ai_keyword}: {response}"
    history_text += f"\n {user_keyword}: {text} \n {ai_keyword}: " 
    if history_summary and related_memory_content and personality:
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=history_text,memo_dates=memo_dates)
    else:
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name,history_text=history_text)
    # print(prompt)
    return prompt,related_memos

import openai
def build_prompt_with_search_memory_llamaindex(history,text,user_memory,user_name,user_memory_index,service_context,api_keys,api_index,meta_prompt,new_user_meta_prompt,data_args,boot_actual_name):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n User：{query}"
    #     history_content += f"\n AI：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]：" 
    memory_search_query = f'和问题：{text}。最相关的内容是：' if data_args.language=='cn' else f'The most relevant content to the question "{text}" is:'
    if user_memory_index:
        related_memos = user_memory_index.query(memory_search_query,service_context=service_context)
    
        retried_times,count = 10,0
        
        while not related_memos and count<retried_times:
            try:
                related_memos = user_memory_index.query(memory_search_query,service_context=service_context)
            except Exception as e:
                print(e)
                api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                openai.api_key = api_keys[api_index]

        related_memos = related_memos.response
    else:
        related_memos = ''
    if "overall_history" in user_memory:
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if data_args.language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
        related_memory_content = f"\n{str(related_memos).strip()}\n"
    else:
        history_summary = ''
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
    
    if related_memos:
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name)
    else:
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name)
    return prompt,related_memos
