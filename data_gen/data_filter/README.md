# Data Filter

## Running IC9600
You need to first clone the repository: [https://github.com/tinglyfeng/IC9600](https://github.com/tinglyfeng/IC9600) to your local machine.

Then follow the instructions in the [IC9600 Repo](https://github.com/tinglyfeng/IC9600) to install the dependencies and download the checkpoint to `/path/to/your/IC9600/checkpoint/ck.pth`, which is required by the `gene_modified_multi.py` script.
```python
model = ICNet()
model.load_state_dict(torch.load('./checkpoint/ck.pth', map_location=torch.device('cpu')))
model.eval()
model.to(device)
```
Then you need to copy the `gene_modified_multi.py` script to `/path/to/your/IC9600/gene_modified_multi.py` and run the following command, this will run the inference on all of the images in the `IMAGE_DIR` directory and save the results to the `OUTPUT_DIR` directory:
```bash
# run ic
cd /path/to/your/IC9600
torchrun --nproc-per-node=$GPU_RAM ./gene_modified_multi.py \
--input $IMAGE_DIR \
--output $OUTPUT_DIR
echo "ic done"

# sort the results by score, this will generate a _sorted.csv file in the OUTPUT_DIR directory
cd data_filter
python ./sort_by_score.py --jsonl_path $OUTPUT_DIR/overall/final_output.jsonl
```

## Running RAM++
You need to first clone the repository: [https://github.com/recognize-anything/recognize-anything](https://github.com/recognize-anything/recognize-anything) to your local machine and then `pip install -e .` to install the dependencies. The `add_ram.py` will need to import modules like `from ram.models import ram_plus`.
```bash
torchrun --nproc-per-node=$GPU_RAM ./add_ram.py \
--input-csv $OUTPUT_DIR/overall/final_output_sorted.csv \
--pretrained /path/to/your/recognize-anything/pretrained/ram_plus_swin_large_14m.pth \
--image-dir $IMAGE_DIR \
--output-dir $OUTPUT_DIR
echo "ram done"
```

## Extra Notes
Main logic of the two scripts are in the `gene_modified_multi.py` and `add_ram.py` scripts. You can modify the code to fit your own needs. And you can try to use any or both of them to filter the images. For example, you can filter the images by the IC score and the RAM cnt through setting the threshold for each of them for your own needs.