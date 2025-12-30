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
        assert os.getenv('ZHIPU_AI_API_KEY') is not None, \
            "Please set the environment variable ZHIPU_AI_API_KEY"
        client = ZhipuAI(api_key=os.getenv('ZHIPU_AI_API_KEY'))
    
    elif cfg.model.startswith("MOONSHOT"):
        from openai import OpenAI
        assert os.getenv('MOONSHOT_API_KEY') is not None, \
            "Please set the environment variable MOONSHOT_API_KEY"
        client = OpenAI(
            api_key=os.getenv('MOONSHOT_API_KEY'),
            base_url="https://api.moonshot.cn/v1"
        )

    elif cfg.model.startswith("qwen"):
        from openai import OpenAI
        # assert os.getenv('QWEN_API_KEY') is not None, \
        #     "Please set the environment variable QWEN_API_KEY"
        client = OpenAI(
            api_key="sk-92c5a2abed7a4f2c9d86a6555a4ce925", 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    else:
        from openai import OpenAI
        # Default: use local or custom OpenAI-compatible API
        base_url = os.getenv('CUSTOM_API_BASE_URL', 'http://localhost:8000/v1/')
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        

def file_to_string(filename: str) -> str:
    """Read entire file content as a string.
    
    Args:
        filename: Path to the file
        
    Returns:
        File content as string
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def filter_traceback(s: str) -> str:
    """Extract traceback error message from output string.
    
    Args:
        s: Output string that may contain traceback
        
    Returns:
        Traceback message if found, empty string otherwise
    """
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''

def block_until_running(stdout_filepath: str, log_status: bool = False, 
                       iter_num: int = -1, response_id: int = -1) -> None:
    """Block execution until the evaluation process has started writing output.
    
    Args:
        stdout_filepath: Path to the stdout file to monitor
        log_status: Whether to log execution status
        iter_num: Current iteration number for logging
        response_id: Response ID for logging
    """
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status:
                if "Traceback" in log:
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


def extract_code_from_generator(content: str) -> str:
    """Extract Python code from LLM response.
    
    Args:
        content: LLM response text
        
    Returns:
        Extracted Python code string or None if no valid code found
    """
    # Try to extract code from markdown code block
    pattern_code = r'```python(.*?)```'
    code_match = re.search(pattern_code, content, re.DOTALL)
    code_string = code_match.group(1).strip() if code_match else None
    
    # Fallback: extract function definition manually
    if code_string is None:
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
            code_string = '\n'.join(lines[start:end + 1])
    
    # Validate extracted code
    if code_string is None:
        return None
    
    if "return" not in code_string:
        return None
    
    # Add missing import statements
    if "np" in code_string and "import numpy" not in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string and "import torch" not in code_string:
        code_string = "import torch\n" + code_string
    
    return code_string


def filter_code(code_string: str) -> str:
    """Remove function signature and import statements from code.
    
    Keeps only the function body up to and including the return statement.
    
    Args:
        code_string: Python code string
        
    Returns:
        Filtered code containing only the function body
    """
    if code_string is None:
        return ""
    
    lines = code_string.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip function definition, imports
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        # Include return statement and stop
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        # Include function body
        else:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def get_heuristic_name(module, possible_names: list[str]) -> str:
    """Find the first function name from possible_names that exists in module.
    
    Args:
        module: Python module to search
        possible_names: List of possible function names
        
    Returns:
        Name of the first matching function found, or None
    """
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
    return None