import json
import os
from openai import OpenAI
import sys
import sys
# from IPython.display import display, HTML

# path_to_add = ""
# if path_to_add not in sys.path:
#     sys.path.append(path_to_add)
# print(sys.path)
from data_gen.utils.log import get_logger

logger = get_logger(__name__)


def close_proxy():
    os.environ["https_proxy"] = ""
    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""


openai_api_key_qwen2_5_72b = ""
openai_api_base_qwen2_5_72b = ""

client_qwen2_5_72b = OpenAI(
    api_key=openai_api_key_qwen2_5_72b,
    base_url=openai_api_base_qwen2_5_72b,
)

close_proxy()


# return the key of image
def get_real_image_key(item):
    for key in [
        "image",
        "filename",
        "img_name",
        "image_name",
        "image_local_name",
        "image_path",
        "image_local_path",
        "img",
        "img_path",
        "img_local_path",
        "img_local_name",
    ]:
        if item.get(key, None) is not None:
            return key
    return None


def run_once_without_image_qwen2_5_72b(pt, retry=3):
    close_proxy()
    global client_qwen2_5_72b
    messages = [{"role": "user", "content": [{"type": "text", "text": pt}]}]

    num_retries = 0
    while num_retries < retry:
        try:
            chat_response = client_qwen2_5_72b.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                temperature=0.5,
                top_p=0.8,
                max_tokens=1024,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            # print("Chat response:", chat_response)
            # get text content from response
            message_content = chat_response.choices[0].message.content
            return message_content
        except Exception as e:
            logger.error(f"Error: {e}")
            num_retries += 1
    raise RuntimeError(
        "Calling OpenAI API failed after retrying for " f"{retry} times."
    )


def fix_json_resp_with_qwen2_5_72b(resp):
    close_proxy()
    # try to fix the response with model
    pt = f"""\
## Task Description
Please fix the response and return the correct json format. 
1. For example, original response contains comments, you should remove them.
2. If the response contains a json and other text, you should extract only the json part and fix it to the correct json format if necessary.
Your output should be a json with no other text.

## Response(need to be fixed)
{resp}
"""
    logger.info(f"Start fixing gpt_resp with qwen2_5_72b")
    resp = run_once_without_image_qwen2_5_72b(pt)
    return resp


# resp = run_once_without_image_qwen2_5_72b("Hello, world!")
# print(resp)


def make_prompt(instruction, constraints_list):
    # constraints may be empty
    # "constraints": [{"key": "not_mention", "value": "construction"}]
    # print("constraints_list: ", constraints_list)
    # print("type of constraints_list: ", type(constraints_list))
    # with index 1. 2. 3. ...
    # breakpoint()
    constraints_list_str = ""
    for i, constraint in enumerate(constraints_list):
        if isinstance(constraint, str):
            constraints_list_str += f"{i+1}. {constraint}\n"
        else:
            constraints_list_str += f"{i+1}. {constraint['key']}: {constraint['value']}\n"  # type: ignore
    prompt = f"""\
## Task Description
{instruction}

## Constraints
You must strictly follow the constraints below:
{constraints_list_str}
"""
    return prompt

def make_prompt_v2(instruction, constraints_list):
    # constraints may be empty
    # "constraints": [{"key": "not_mention", "value": "construction"}]
    # print("constraints_list: ", constraints_list)
    # print("type of constraints_list: ", type(constraints_list))
    # with index 1. 2. 3. ...
    # breakpoint()
    # constraints_list_str = ""
    pt = instruction
    for i, constraint in enumerate(constraints_list):
        pt += " " + constraint["value"]
    return pt


# transform list to string with index 1. 2. 3. ...
def list_to_str(l):
    return "\n".join([f"{i+1}. {x}" for i, x in enumerate(l)])


# def print_colored(text, color="green"):
#     if "ipykernel" in sys.modules:
#         # run in Jupyter Notebook or VSCode Jupyter
#         display(HTML(f"<span style='color: {color};'>{text}</span>"))
#     else:
#         # use colorama in terminal
#         from colorama import Fore

#         print(Fore.GREEN + text)


# output_files: list of output file paths (jsonl format)
# final_output: the file to merge all output files (jsonl format)
def merge_output_files(output_files, final_output):
    with open(final_output, "w") as fout:
        for file in output_files:
            with open(file, "r") as fin:
                for line in fin:
                    fout.write(line)


def close_proxy():
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""


# load progress file
def load_progress(progress_file) -> set[str]:
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))  # return processed file set
    return set()


# save progress to file
# processed_imgs: set of processed file names
# progress_file: the file path to save progress
def save_progress(progress_file, processed_imgs):
    with open(progress_file, "w") as f:
        json.dump(list(processed_imgs), f)
