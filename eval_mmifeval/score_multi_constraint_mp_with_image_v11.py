import argparse
import json
import os
from pprint import pprint
from vlmeval.api import OpenAIWrapper

# from colorama import Fore, Back, Style, init
from data_gen.utils.tools import (
    get_real_image_key,
    close_proxy,
)
from data_gen.utils.function_and_compare import *
from data_gen.utils.log import get_logger
from vlmeval.utils.mp_util import track_progress_rich
from vlmeval.smp import *
import re


def generate_eval_pt(constraints, prediction):
    constraints_str = "\n".join(
        [
            f"Constraint_{i+1}: {constraint['value']}"
            for i, constraint in enumerate(constraints)
        ]
    )
    pt = f"""\
Your task is to evaluate whether the response from an AI assistant adheres to all of the given constraints. \
Please follow the requirements below to make the judgment:
1. Be strict and consistent in your assessment.
2. You should refer to the content of image to make the judgment.
3. For each constraint, if the response fails to fully meet the constraint, give it a score of 0. Otherwise, give it a score of 1.

<start of response>
{prediction}
<end of response>

<start of constraint list>
{constraints_str}
<end of constraint list>

You must evaluate and provide an explanation for each constraint listed, ensuring no constraint is omitted. \
At the end, summarize the scores for all constraints in one sentence.

Your output should strictly follow the format below:
Judgement: ...
Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, Score of constraint_3: x/1, ..., Score of constraint_n: x/1.
"""
    return pt


def generate_eval_pt_vision(question, prediction, ground_truth):
    pt = f"""\
You are an expert evaluator. Your task is to extract the answer from the model output and compare it with the ground truth list to determine whether the model answer covers all the points in the ground truth list. \
The ground truth list is provided as a JSON array of strings, and the model answer is a text string. \
An answer is considered correct if every element from the ground truth list appears in the model answer (substring matching is acceptable). \
The order does not matter. \

Your response should only be 'right' if the model answer fully covers the ground truth, or 'wrong' if it does not. \
Do not provide any additional commentary.

Question: {question}
Response from the model: {prediction}
Ground Truth List: {ground_truth}
"""
    return pt


def generate_cmp_pt(constraint, pred_with_constraint, pred_without_constraint):
    pt = f"""\
You are an expert in judging whether the respone follow the given constraint. Your task is to assess whether the model's response satisfies the given constraint and return True or False. I will provide you with the constraint and the model's response under this constraint. To assist with your evaluation, I will also provide you with the model's response to the same question without the constraint.

<start of constraint>
{constraint}
<end of constraint>

<start of response under the constraint>
{pred_with_constraint}
<end of response under the constraint>

<start of response without the constraint>
{pred_without_constraint}
<end of response without the constraint>

**Please follow the steps below to evaluate**:
Step 1. Compare the model's response under the constraint with its response without the constraint. If you believe these two answers are very similar, it means the model has not fully considered the impact of the constraint on the answer. Please return False.
Step 2. Compare the model's response under the constraint with the content of the constraint. If you believe the model's response does not meet the requirements specified in the constraint, return False. Otherwise, if the response effectively satisfies the constraint, return True.

Start by briefly explaining your reasoning based on the above steps. At the end, provide a one-sentence summary of your evaluation.

Your output must strictly follow this format:  
Reasoning: ...  
Summary: "True" / "False".
"""
    return pt


logger = get_logger(__name__)
logger.propagate = False

parser = argparse.ArgumentParser(description="Inference on a benchmark dataset")
parser.add_argument("--model_name", type=str, default=None, help="Model name")
parser.add_argument("--bench_name", type=str, default=None, help="Benchmark name")
parser.add_argument("--inference_file", type=str, default=None, help="Inference file")
parser.add_argument("--project_dir", type=str, default=None, help="Project directory")
args = parser.parse_args()

# print(prompts)

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["OPENAI_API_BASE"] = ""

close_proxy()

project_dir = args.project_dir

# read input file
if args.inference_file is not None:
    input_file = args.inference_file
elif args.model_name is not None and args.bench_name is not None:
    input_file = f"{project_dir}/eval_results/{args.model_name}/{args.bench_name}/processed/{args.model_name}_{args.bench_name}.jsonl"
else:
    raise ValueError(
        "Either inference_file or model_name and bench_name must be provided"
    )

img_root = f"{project_dir}/data"

with open(input_file, "r") as f:
    data_all = f.readlines()

# real_image_key = get_real_image_key(json.loads(data_all[0]))

data_all = [json.loads(line) for line in data_all]
# split data_all into main_data and aux_data
main_data = []
aux_data = []
for line in data_all:
    if line.get("infer_type", None) == "main":
        main_data.append(line)
    else:
        aux_data.append(line)

# main_data = [json.loads(line) for line in main_data]
# aux_data = [json.loads(line) for line in aux_data]

# process aux_data to a dict
aux_data_dict = {}
for line in aux_data:
    assert line["infer_type"] == "aux_cmp_gpt"
    del_cons = line["del_cons"]
    if line["id"] not in aux_data_dict:
        aux_data_dict[line["id"]] = {}
    aux_data_dict[line["id"]][del_cons] = line["prediction"]

real_image_key = get_real_image_key(main_data[0])

# judge_model_name = "gpt-4o-mini"
if main_data[0].get("tag", None) == "P-Level":
    judge_model_name = "gpt-4o-mini"
else:
    judge_model_name = "gpt-4o-2024-11-20"
gpt = OpenAIWrapper(
    judge_model_name,
    temperature=0,
    max_tokens=4096,
    img_detail="high",
    img_size=-1,
    timeout=300,
)
# with image
# params_all = [
#     (generate_eval_pt(item["instruction"], item["constraints"], item["prediction"]), img_root + "/" + item[real_image_key])
#     for item in data_all
# ]

# without image
# if main_data[0].get('tag', None) == 'vision':
#     params_all = [
#         (generate_eval_pt_vision(item["question"], item["prediction"], item["answer"]))
#         for item in data_all
#     ]
# else:
#     params_all = [
#         (generate_eval_pt(item["instruction"], item["constraints"], item["prediction"]))
#         for item in data_all
#     ]

# merge
# item as a param
# transfer need to be str, dict will be error
# like: TypeError: judge_one_item() got an unexpected keyword argument 'id'
# transform dict to str
params_all = [json.dumps(item) for item in main_data]

output_file = input_file.replace(".jsonl", f"_{judge_model_name}.jsonl")

suffix = output_file.split(".")[-1]
tmp_file = output_file.replace(f".{suffix}", f"_temp.pkl")
# pkl stores the key-value pairs of the function return value
# for example:
# 'allava/allava_vflan/images/images_191task_1k/MEMOTION+sentiment_detection_749_image_6095.jpg': (0, '{\n    "constraint_1": "1/1",\n    "constraint_2": "1/1",\n    "constraint_3": "0/1",\n    "constraint_4": "0/1",\n    "constraint_5": "0/1",\n    "total_score": "2/5"\n}', <Response [200]>)
# The file name corresponds to a tuple, which is the return value of the *function*
# You can choose the key you want (for example, id/real_image_key 's value)
indices_all = [line["id"] for line in main_data]


# input: pt, image_path
# return: (ret_code, ans, response)
def run_once_with_image(pt, image_path, retry=5):
    global gpt
    num_retries = 0
    message = []
    text = {"type": "text", "value": pt}
    image = {"type": "image", "value": image_path}
    message.append(text)
    message.append(image)
    # return gpt4o.generate_inner(message)
    # Boost Code Robustness, 2024.11.11
    while num_retries < retry:
        ret_code, ans, response = gpt.generate_inner(message)
        if ret_code == 0:
            return ret_code, ans, response
        else:
            num_retries += 1
    return ret_code, ans, response


# input: pt
# return: (ret_code, ans, response)
def run_once_without_image(pt, retry=5):
    global gpt
    num_retries = 0
    message = []
    text = {"type": "text", "value": pt}
    message.append(text)
    while num_retries < retry:
        ret_code, ans, response = gpt.generate_inner(message)
        if ret_code == 0:
            return ret_code, ans, response
        else:
            num_retries += 1
            logger.info(f"Start retry {num_retries}.")
    return ret_code, ans, response


def calculate_score(json_resp):
    score = 0
    for key, value in json_resp.items():
        if value == "true":
            score += 1
    return score


# extract score from gpt_resp
# format: Score of instruction: x/1, Score of constraint_1: y/1, Score of constraint_2: z/1, ..., Score of constraint_n: w/1.
# return: score_dict {'instruction': x/1, 'constraint_1': y/1, 'constraint_2': z/1, ..., 'constraint_n': w/1}
def extract_score_from_direct_gpt_resp(raw_score):
    # Define regular expression patterns (updated to handle underscores in constraint names)
    score_pattern = re.compile(
        r"Score\s+of\s+([a-zA-Z0-9_\-]+):\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE
    )

    # Clean the raw score to remove unnecessary symbols (e.g., newlines, multiple spaces)
    cleaned_score = re.sub(r"\s+", " ", raw_score).strip()  # Normalize whitespace
    # delete all the '*'
    cleaned_score = re.sub(r"\*", "", cleaned_score)

    # Find all individual component scores
    score_matches = score_pattern.findall(cleaned_score)

    # If no valid score matches found, print and raise an exception
    if not score_matches:
        print(f"raw_score:\n{raw_score}")
        raise ValueError("raw_score format is incorrect, cannot parse scores")

    score_dict = {}

    # Parse each component score
    for match in score_matches:
        component_name = (
            match[0].strip().lower()
        )  # Component name, converted to lowercase
        component_name = component_name.replace(" ", "_")
        numerator = int(match[1])  # Numerator
        denominator = int(match[2])  # Denominator
        score = numerator / denominator  # Calculate the score
        score_dict[component_name] = score  # Store it in the dictionary

    return score_dict


# extract score from gpt_resp
# format: right or wrong
# return: score
# def extract_cmp_judge_score(response_text):
#     # Step 1: Find the last occurrence of 'summary:'
#     summary_idx = response_text.lower().rfind("summary")
#     if summary_idx == -1:
#         raise ValueError("No 'summary' found in response.")

#     # Step 2: Slice the string after 'summary:' and extract value
#     after_summary = response_text[summary_idx + len("summary") :]

#     # Match true/false ignoring markdown and formatting
#     match = re.search(r"\b(true|false)\b", after_summary, re.IGNORECASE)
#     if match:
#         value = match.group(1).lower()
#         return 1 if value == "true" else 0

#     raise ValueError("No valid 'True' or 'False' found after 'summary'.")

# extract score from gpt_resp
# format: right or wrong
# return: score
def extract_score_from_vision_gpt_resp(raw_score):
    if raw_score == "right":
        return 1
    elif raw_score == "wrong":
        return 0
    else:
        # 尝试在整个字符串中匹配"right"或"wrong"，注意大小写
        if re.search(r"right", raw_score, re.IGNORECASE):
            return 1
        elif re.search(r"wrong", raw_score, re.IGNORECASE):
            return 0
        else:
            raise ValueError("raw_score format is incorrect, cannot parse scores")

# extract score from gpt_resp
# format: True or False
# return: score
def extract_score_from_cmp_gpt_resp(response_text):
    # Step 1: Find the last occurrence of 'summary:'
    summary_idx = response_text.lower().rfind("summary")
    if summary_idx == -1:
        raise ValueError("No 'summary' found in response.")

    # Step 2: Slice the string after 'summary:' and extract value
    after_summary = response_text[summary_idx + len("summary") :]

    # Match true/false ignoring markdown and formatting
    match = re.search(r"\b(true|false)\b", after_summary, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        return 1 if value == "true" else 0

    raise ValueError("No valid 'True' or 'False' found after 'summary'.")


# judge one item
# return: (ret_code: 0 means success, 1 means fail, msg: "success" or "fail reason", score_dict: score_dict)
def judge_one_item(item):
    item = json.loads(item)
    if item.get("tag", None) == "P-Level":
        pt = generate_eval_pt_vision(
            item["question"], item["prediction"], item["answer"]
        )
        ret_code, gpt_resp, full_resp = run_once_without_image(pt)
        if ret_code != 0:
            logger.error(
                f"\nItem:\n{item}\ngpt_resp:\n{gpt_resp}\nfull_resp:\n{full_resp}"
            )
            return 1, "Vision data, fail in get gpt_resp", {}
        try:
            score = extract_score_from_vision_gpt_resp(gpt_resp)
            return (
                0,
                "success",
                {
                    "total_score": score,
                    "gpt_resp": gpt_resp,
                },
            )
        except Exception as e:
            logger.error(
                f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}\nfull_resp:\n{full_resp}"
            )
            return 1, "Vision data, fail in extract score", {}
    else:  # process text data
        # split into direct_gpt and other
        # direct_gpt can be processed in batch
        # other needs to be processed one by one
        # breakpoint()
        constraint_direct_gpt = []
        constraint_other = []
        for constraint in item["constraints"]:
            method = constraint["judge"]["method"]
            if method == "direct_gpt":
                constraint_direct_gpt.append(constraint)
            else:
                constraint_other.append(constraint)
        score_dict = {}
        # 1. process direct_gpt: if there is no direct_gpt, instruction is also needed
        if len(constraint_direct_gpt) > 0:
            pt_direct_gpt = generate_eval_pt(constraint_direct_gpt, item["prediction"])
            # logger.info(f"pt_direct_gpt:\n{pt_direct_gpt}")
            # ret_code, gpt_resp, full_resp = run_once_without_image(pt_direct_gpt)
            image_path = img_root + "/" + item[real_image_key]
            ret_code, gpt_resp, full_resp = run_once_with_image(
                pt_direct_gpt, image_path
            )
            # logger.info(f"gpt_resp for direct_gpt:\n{gpt_resp}")
            if ret_code != 0:
                logger.error(
                    f"\nItem:\n{item}\npt_direct_gpt:\n{pt_direct_gpt}\ngpt_resp:\n{gpt_resp}\nfull_resp:\n{full_resp}"
                )
                return 1, "None vision data, direct_gpt, fail in get gpt_resp", {}
            try:
                direct_gpt_score_dict = extract_score_from_direct_gpt_resp(gpt_resp)
                score_dict["gpt_resp_direct_gpt"] = gpt_resp
                # 不传instruction了
                # score_dict["instruction"] = direct_gpt_score_dict["instruction"]
                for i, constraint in enumerate(constraint_direct_gpt):
                    score_dict[constraint["key"]] = direct_gpt_score_dict[
                        f"constraint_{i+1}"
                    ]
            except Exception as e:
                logger.error(
                    f"\nError:\n{e}\nItem:\n{item}\npt_direct_gpt:\n{pt_direct_gpt}\ngpt_resp:\n{gpt_resp}\nfull_resp:\n{full_resp}"
                )
                return 1, "None vision data, direct_gpt, fail in extract score", {}
        # print log after success in direct_gpt
        # logger.info(f"Success: direct_gpt")
        # logger.info(f"score_dict:\n{score_dict}")
        # 2. process rule_based
        for constraint in constraint_other:
            if constraint["judge"]["method"] == "rule_based":
                # call function according to constraint["judge"]["verify_funcs"]
                # maybe a list of function names (str)
                # func in function_and_compare.py
                # example: {"method": "rule_based", "verify_funcs": [{"func": "check_whether_response_paragraph_number_in_range", "params": [3, 3]}]}}
                score = 1.0
                # breakpoint()
                for func_dict in constraint["judge"]["verify_funcs"]:
                    func = globals()[func_dict["func"]]
                    # use * to unpack the list, ** is used for dict
                    judge_result = func(item["prediction"], *func_dict["params"])
                    # breakpoint()
                    if not judge_result:  # False -> score = 0
                        score = 0.0
                        break
                # breakpoint()
                score_dict[constraint["key"]] = score
        # breakpoint()
        # logger.info(f"Success: rule_based")
        # logger.info(f"score_dict:\n{score_dict}")
        # 3. process direct_gpt
        for constraint in constraint_other:
            if constraint["judge"]["method"] == "cmp_gpt":
                del_cons_prediction = aux_data_dict[item["id"]][constraint["key"]]
                pt = generate_cmp_pt(
                    constraint["value"], item["prediction"], del_cons_prediction
                )
                ret_code, gpt_resp, full_resp = run_once_without_image(pt)
                if ret_code != 0:
                    logger.error(
                        f"\nItem:\n{item}\npt\ngpt_resp:\n{gpt_resp}\nfull_resp:\n{full_resp}"
                    )
                    return 1, "None vision data, cmp_gpt, fail in get gpt_resp", {}
                try:
                    score = extract_score_from_cmp_gpt_resp(gpt_resp)
                    score_dict[constraint["key"]] = score
                    score_dict[f"gpt_resp_cmp_gpt_{constraint['key']}"] = gpt_resp
                except Exception as e:
                    logger.error(
                        f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}\nfull_resp:\n{full_resp}"
                    )
                    return 1, "None vision data, cmp_gpt, fail in extract score", {}
        # logger.info(f"Success: cmp_gpt")
        # logger.info(f"score_dict:\n{score_dict}")
        # Finally return score_dict
        # add total_score
        total_score = 0.0
        cnt = 0
        for key, value in score_dict.items():
            if key.startswith("gpt_resp_"):
                continue
            total_score += value
            cnt += 1
        score_dict["total_score"] = total_score / cnt
        logger.info(f"score_dict:\n{score_dict}")
        # breakpoint()
        return 0, "success", score_dict


def score_once():
    ans = {}
    if os.path.exists(tmp_file):
        ans = load(tmp_file)
        # ans is a dict
        logger.info(f"Loaded {len(ans)} data from {tmp_file}")

    tups = [x for x, i in zip(params_all, indices_all) if i not in ans]
    indices = [i for i in indices_all if i not in ans]
    logger.info(f"Temp file: {tmp_file}")
    logger.info(f"Total {len(indices_all)} data to be scored")
    logger.info(f"Left {len(indices)} data in this round to be scored")
    # breakpoint()
    nproc = 8
    # nproc = 1
    if len(indices) > 0:
        results = track_progress_rich(
            judge_one_item,
            tups,
            nproc=nproc,
            chunksize=nproc,
            keys=indices,
            save=tmp_file,
        )
    # load new ans
    ans = load(tmp_file)

    # only process the data not in output_file
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            indices_done = [json.loads(line)["id"] for line in f.readlines()]
    else:
        indices_done = []
    data_left = [line for line in main_data if line["id"] not in indices_done]
    logger.info(f"Output file: {output_file}")
    logger.info(f"Left {len(data_left)} data in this round to be processed")

    # process function return value and write to output_file
    with open(output_file, "a") as f:
        from tqdm import tqdm

        for line in tqdm(data_left):
            ret_code, msg, score_dict = ans[line["id"]]
            # if ret_code is not 0, skip this data (it means the data is not valid)
            if ret_code != 0:
                logger.error(f"\nMsg:\n{msg}\nItem_id:\n{line['id']}")
                del ans[line["id"]]
                continue
            # directly get score from ans
            line["score"] = score_dict
            f.write(json.dumps(line) + "\n")
            f.flush()
        # [Important] ans has been changed so we need to save it
        dump(ans, tmp_file)


# check if all data have been scored
num_retries = 0
while num_retries < 10:
    if os.path.exists(output_file):
        with open(output_file, "r") as f2:
            data2 = f2.readlines()
        if len(main_data) == len(data2):
            logger.info(f"All data have been scored.")
            break
    score_once()
    num_retries += 1


# generate summary
import pandas as pd

result_jsonl = output_file
output_xlsx = output_file.replace(".jsonl", "_summary.xlsx")

try:
    read_data = pd.read_json(output_file, lines=True)
    logger.info(f"Successfully read {output_file}")
except ValueError as e:
    logger.error(f"Error reading JSONL file: {e}")
    raise

score_sum = 0
for (
    _,
    line,
) in read_data.iterrows():  # use iterrows() to iterate over the rows of the DataFrame
    score_sum += line["score"]["total_score"]
accuracy = score_sum / len(read_data)
print(f"total_score: {score_sum}, len: {len(read_data)}, accuracy: {accuracy}")

# Create overall summary DataFrame
summary_overall = pd.DataFrame(
    {
        "model": [args.model_name],
        "bench": [args.bench_name],
        "score_sum": [score_sum],
        "len": [len(read_data)],
        "accuracy": [accuracy * 100.0],
    }
)

# save to jsonl
with open(output_file.replace(".jsonl", "_summary.jsonl"), "w") as f:
    f.write(summary_overall.to_json(orient="records"))
