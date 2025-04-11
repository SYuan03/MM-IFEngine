import argparse
import os
import json
import shutil
from vlmeval.config import supported_VLM
from data_gen.utils.tools import get_real_image_key, make_prompt, make_prompt_v2
from data_gen.utils.log import get_logger
import torch
from tqdm import tqdm

logger = get_logger(__name__)

parser = argparse.ArgumentParser(description="Inference on a benchmark dataset")
parser.add_argument("--model_name", type=str, default='Qwen2-VL-72B-Instruct', help="Model name")
parser.add_argument("--bench_name", type=str, default='v3', help="Benchmark name")
parser.add_argument("--current_time", type=str, default='20241217_103803', help="Current time")
parser.add_argument("--project_dir", type=str, default='', help="Project directory")
args = parser.parse_args()

def get_rank_and_world_size():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return local_rank, world_size

def setup_distributed(rank, world_size):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

local_rank, world_size = get_rank_and_world_size()
print(f"local_rank: {local_rank}, world_size: {world_size}")
setup_distributed(local_rank, world_size)

# Set model
model_name = args.model_name
bench_name = args.bench_name
model = supported_VLM[model_name]()
img_root = f"{args.project_dir}/data"
input_file = f'{args.project_dir}/data/MM-IFEval/{bench_name}.jsonl'
current_time = args.current_time
output_dir = f'{args.project_dir}/eval_results/{model_name}/{bench_name}/{current_time}/'
os.makedirs(output_dir, exist_ok=True)

# Read data and json load
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

real_image_key = get_real_image_key(data[0])
logger.info(f"real_image_key: {real_image_key}")

# Remove already processed data from all ranks
index_set = set()
for i in range(world_size):
    temp_output_file = os.path.join(output_dir, f"output_rank_{i}.jsonl")
    if os.path.exists(temp_output_file):
        print(f"Exist output file for rank {i}: {temp_output_file}, resuming...")
        with open(temp_output_file, 'r') as f:
            for line in f:
                try:
                    data_temp = json.loads(line)
                    index_set.add(data_temp["index"])
                except Exception as e:
                    print(f"Error: {e}")
                    continue
output_file = os.path.join(output_dir, f"output_rank_{local_rank}.jsonl")

data = [line for line in data if line["index"] not in index_set]
logger.info(f"Remaining data for all ranks: {len(data)}")

# wait all ranks to finish
torch.distributed.barrier()

# Distribute data among ranks
data = [line for i, line in enumerate(data) if i % world_size == local_rank]
logger.info(f"Data for local_rank {local_rank}: {len(data)}")

# Write data to a single output file for each rank
with open(output_file, 'a') as f:
    for line in tqdm(data, desc=f"Processing data (Rank {local_rank})"):
        if line.get('tag', None) == 'P-Level':
            prompt = line['question']
        else:
            prompt = make_prompt_v2(line['instruction'], line.get('constraints', []))
        logger.info(f"prompt: {prompt}")
        img_path = os.path.join(img_root, line[real_image_key])
        result = model.generate([img_path, prompt])
        line['prediction'] = result

        f.write(json.dumps(line, ensure_ascii=False) + '\n')
        f.flush()

# Barrier to ensure all ranks finish
torch.distributed.barrier()

# Merge output files (only by rank 0)
if local_rank == 0:
    final_output_file = os.path.join(output_dir, f"{model_name}.jsonl")
    with open(final_output_file, 'w') as final_output:
        for rank in range(world_size):
            rank_output_file = os.path.join(output_dir, f"output_rank_{rank}.jsonl")
            if os.path.exists(rank_output_file):
                with open(rank_output_file, 'r') as f:
                    shutil.copyfileobj(f, final_output)

    # Copy to processed directory
    processed_dir = os.path.join(output_dir, "../processed")
    os.makedirs(processed_dir, exist_ok=True)
    shutil.copy(
        final_output_file,
        os.path.join(processed_dir, f"{model_name}_{bench_name}.jsonl"),
    )
    logger.info(f"Copied output file to ../processed and renamed it to {model_name}_{bench_name}.jsonl")

