# model_loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import BitsAndBytesConfig

def get_model():

    _model = None
    _tokenizer = None   
    if _model is None or _tokenizer is None: 
        mode_name_or_path = os.environ.get("BASE_LLM_MODEL_PATH", "THUDM/glm-4-9b-chat")
        # 从预训练的模型中获取tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=True, trust_remote_code=True)
        
        _tokenizer.pad_token = _tokenizer.eos_token

        # 从预训练的模型中获取模型，并设置模型参数jdfh
        #quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 启用 INT8 量化
        _model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16,trust_remote_code=True).to('cuda:0').eval()#,device_map="auto"

        # _model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        # _model.generation_config.pad_token_id = _model.generation_config.eos_token_id

    return _tokenizer, _model

def generate_respond(tokenizer, model,prompt,gen_kwargs):

    input_ids = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_ids], return_tensors="pt",padding=True).to(model.device)

    generated_ids = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        **gen_kwargs
    )

    # 生成模型响应
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
