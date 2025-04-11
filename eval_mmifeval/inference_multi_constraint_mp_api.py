import argparse
import os
import json
import shutil
import threading
import time
from vlmeval.config import supported_VLM
from data_gen.utils.tools import get_real_image_key, make_prompt_v2
from data_gen.utils.log import get_logger
import torch
from tqdm import tqdm

logger = get_logger(__name__)

parser = argparse.ArgumentParser(description="Inference on a benchmark dataset")
parser.add_argument("--model_name", type=str, default='Qwen2-VL-72B-Instruct', help="Model name")
parser.add_argument("--bench_name", type=str, default='C-Level', help="Benchmark name")
parser.add_argument("--img_root", type=str, default='', help="Image root name")
parser.add_argument("--current_time", type=str, default='20202020_202020', help="Current time")
parser.add_argument("--num_threads", type=int, default=1, help="Number of threads")
parser.add_argument("--project_dir", type=str, default='', help="Project directory")

args = parser.parse_args()

if args.model_name.startswith('Claude'):
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# result = model.generate([img_path, prompt])
# line['prediction'] = result

def thread_worker(
    image_root_dir,
    flt_data,
    output_file,
    real_image_key,
    model,
):
    with tqdm(
        total=len(flt_data), desc=f"Thread-{threading.current_thread().name}"
    ) as pbar:
        with open(output_file, "a") as f:  # append mode
            for i in range(len(flt_data)):
                data = flt_data[i]
                filename = data[real_image_key]
                image_path = os.path.join(image_root_dir, filename)
                # logger.info(f"image_root_dir: {image_root_dir}")

                # generate prompt
                # todo
                if data.get('tag', None) == 'P-Level':
                    prompt = data['question']
                else:
                    prompt = make_prompt_v2(data['instruction'], data.get('constraints', []))
                # print(f"image_path: {image_path}, prompt: {prompt}")
                # logger.info(f"image_path: {image_path}, prompt: {prompt}")
                result = model.generate([image_path, prompt])
                data["prediction"] = result
                logger.info(f"filename:\n{filename}, prediction:\n{result}")
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(1)

def main():
    # Set model
    model_name = args.model_name
    bench_name = args.bench_name
    num_threads = args.num_threads
    project_dir = args.project_dir
    model = supported_VLM[model_name]()
    img_root = f"{project_dir}/data"
    input_file = f'{project_dir}/data/MM-IFEval/{bench_name}.jsonl'
    current_time = args.current_time
    output_dir = f'{project_dir}/eval_results/{model_name}/{bench_name}/{current_time}/'
    os.makedirs(output_dir, exist_ok=True)

    # Read data and json load
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    real_image_key = get_real_image_key(data[0])
    logger.info(f"real_image_key: {real_image_key}")
    num_threads = args.num_threads  # 根据 CPU 核心数动态调整
    threads = []
    output_files = []

    # <Split data to threads>
    # split df into num_threads chunks, more equally
    # thread_data_map is a dict, key is thread index, value is data number of this thread
    thread_data_map = {}
    chunk_size = len(data) // num_threads
    logger.info(f"chunk_size: {chunk_size}")
    # breakpoint()
    if chunk_size == 0:
        remaining_data_num = len(data)
    else:
        remaining_data_num = len(data) % chunk_size
    if num_threads == 1:
        thread_data_map[0] = len(data)
    else:
        for i in range(num_threads):
            thread_data_map[i] = chunk_size
        # if remaining_data_num > 0, then the former thread will have one more data +1
        for i in range(remaining_data_num):
            thread_data_map[i] += 1

    logger.info(f"thread_data_map: {thread_data_map}")
    # breakpoint()

    sum = 0
    # generate threads
    for i in range(num_threads):
        start_idx = sum
        sum += thread_data_map[i]
        end_idx = start_idx + thread_data_map[i]
        data_chunk = data[start_idx:end_idx]
        output_file = os.path.join(output_dir, f"output_rank_{i}.jsonl")
        output_files.append(output_file)
        thread = threading.Thread(
            target=thread_worker,
            args=(
                img_root,
                data_chunk,
                output_file,
                real_image_key,
                model,
            ),
        )
        threads.append(thread)
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    # merge output files
    final_output_file = os.path.join(output_dir, f"{model_name}.jsonl")
    with open(final_output_file, 'w') as final_output:
        for rank in range(num_threads):
            rank_output_file = os.path.join(output_dir, f"output_rank_{rank}.jsonl")
            if os.path.exists(rank_output_file):
                with open(rank_output_file, 'r') as f:
                    shutil.copyfileobj(f, final_output)

    # copy output file to ../processed
    processed_dir = os.path.join(output_dir, "../processed")
    os.makedirs(processed_dir, exist_ok=True)
    shutil.copy(
        final_output_file,
        os.path.join(processed_dir, f"{model_name}_{bench_name}.jsonl"),
    )
    logger.info(
        f"Copied output file to ../processed and renamed it to {model_name}_{bench_name}.jsonl"
    )

if __name__ == "__main__":
    main()