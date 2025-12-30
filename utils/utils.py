import requests
import subprocess
import os
import json
import logging
import concurrent.futures
import time
import re
import inspect


def init_client(cfg):
    global client
    global url
    global data
    if cfg.model.startswith("gpt"):
        import openai
        from openai import OpenAI
        assert os.getenv('OPENAI_API_KEY') is not None, "Please set the environment variable OPENAI_API_KEY"
        # client = OpenAI(api_key=)
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        openai.api_base = "https://api.chatanywhere.tech"
        
    elif cfg.model.startswith("GLM"):
        from zhipuai import ZhipuAI 
        assert os.getenv('ZHIPU_AI_API_KEY') is not None, "Please set the environment variable ZHIPU_AI_API_KEY"
        zhipu_api_key = os.getenv('ZHIPU_AI_API_KEY')
        client = ZhipuAI(api_key=zhipu_api_key)
    
    elif cfg.model.startswith("MOONSHOT"):
        from openai import OpenAI
        # We use llama api here. See the available models at https://docs.llama-api.com/quickstart#available-models
        assert os.getenv('MOONSHOT_API_KEY') is not None, "Please set the environment variable MOONSHOT_API_KEY"
        client = OpenAI(
        api_key = os.getenv('MOONSHOT_API_KEY'),
        base_url = "https://api.moonshot.cn/v1"
        ) 

    elif cfg.model.startswith("qwen"):
        from openai import OpenAI
        # We use llama api here. See the available models at https://docs.llama-api.com/quickstart#available-models
        client = OpenAI(api_key="sk-0964d3e1695e41a38ef8ec2d45d7c762", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # elif cfg.model.startswith("qwen2"):
    #     url = 'http://222.20.98.39:8533/api/chat'
    #     data = {
    #         "model": "qwen2-coder:7b",
    #         "messages": [
    #             {"role": "system", "content": "你是一个代码工程师"},
    #             {"role": "user", "content": " "}
    #         ],
    #         "stream": False
    #     }
    else:
        from openai import OpenAI
        # We use llama api here. See the available models at https://docs.llama-api.com/quickstart#available-models
        base_url = "http://222.20.98.40:8533/v1/"
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:  # 如果存在则会出现"Traceback"
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature):
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    assert isinstance(messages_list, list), "messages_list should be a list."
    if not isinstance(messages_list[0], list):
        messages_list = [messages_list]
    
    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."
    
    if not model.startswith(("gpt")):
        # Transform messages if n > 1
        messages_list *= n
        n = 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = [(n, messages, model, temperature) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            contents.append(c.message.content)       
    return contents

def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """
    for attempt in range(1000):
        try:
            if "gpt" in model:
                response_cur = client.chat.completions.create(model=model, messages=messages, temperature = min(temperature, 1.), n=n)
            else:
                assert n == 1
                if "GLM" in model:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
                else:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
            break
        except Exception as e:
            logging.info(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
    if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()
            
    return response_cur.choices


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None: 
        return None
    
    if "return" not in code_string: # 如果return语句都没有
        return None
    
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    if code_string is None:
        return ""  # Return an empty string if code_string is None
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name