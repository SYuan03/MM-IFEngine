import argparse
import os
import torch
import torch.distributed as dist
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from PIL import Image
import pandas as pd
import math
from tqdm import tqdm  # used to show progress bar

parser = argparse.ArgumentParser(description="RAM+ Inference")
parser.add_argument(
    "--pretrained",
    type=str,
    default="pretrained/ram_plus_swin_large_14m.pth",
    help="path to pretrained model",
)
parser.add_argument("--input-csv", type=str, default="input.csv", help="input csv file")
parser.add_argument("--image-dir", type=str, default="images", help="image directory")
parser.add_argument("--output-dir", type=str, default="output.csv", help="output csv file")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def get_rank_and_world_size():
    rank = int(os.environ.get("LOCAL_RANK", 0))  # LOCAL_RANK represents the GPU rank of the current process
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # WORLD_SIZE represents the total number of GPUs
    return rank, world_size


def setup_distributed(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def add_ram_cnt_distributed(df, model, transform, device, rank, world_size):
    # get the data range for the current rank
    n_samples = len(df)
    per_rank_samples = math.ceil(n_samples / world_size)
    start = per_rank_samples * rank
    end = min(n_samples, per_rank_samples * (rank + 1))
    ram_cnts = []
    ram_en_tags = []

    # show progress bar for the current rank
    for i in tqdm(range(start, end), desc=f"Rank {rank} Progress"):
        filename = df.iloc[i]["filename"]
        image_path = os.path.join(
            args.image_dir, filename
        )
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)

        # Inference with multi-GPU
        with torch.no_grad():
            res = inference(image, model)
        # print(f"Rank {rank} - {i}: {res[0]}")

        # add ram
        ram_en_tags.append(res[0])
        # count ram
        items = res[0].split(" | ")
        ram_cnts.append(len(items))

    df.loc[start : end - 1, "ram_en_tag"] = ram_en_tags
    df.loc[start : end - 1, "ram_cnt"] = ram_cnts

    # save the data processed by each rank
    temp_csv = os.path.join(args.output_dir, f"temp_rank_{rank}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    # only save the columns with int type
    df.iloc[start:end].astype({"ram_cnt": "int"}).to_csv(temp_csv, index=False)
    print(f"Rank {rank} - Saved partial CSV: {temp_csv}")


def merge_csv_files(world_size):
    df_list = []
    for rank in range(world_size):
        temp_csv = os.path.join(args.output_dir, f"temp_rank_{rank}.csv")
        if os.path.exists(temp_csv):
            df = pd.read_csv(temp_csv)
            df_list.append(df)
    # merge all data
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        output_csv = os.path.join(args.output_dir, "ic_ram_output.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        print(f"Saved merged CSV: {output_csv}")


def main():
    rank, world_size = get_rank_and_world_size()
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    transform = get_transform(image_size=384)

    model = ram_plus(
        pretrained=args.pretrained,
        image_size=384,
        vit="swin_l",
    )
    model = model.to(device)
    model.eval()

    input_csv = args.input_csv
    df = pd.read_csv(input_csv)

    add_ram_cnt_distributed(df, model, transform, device, rank, world_size)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_csv_files(world_size)

    cleanup_distributed()


if __name__ == "__main__":
    main()