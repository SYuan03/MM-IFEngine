#!/bin/zsh

# <---- param settings ---->
PROJECT_DIR=
CONDA_ACTIVATE_PATH=
export HF_HOME=
model_bench_pairs=(
    # "Qwen2-VL-7B-Instruct C-Level 8 qwen_vl HF"
    # "Qwen2-VL-7B-Instruct P-Level 8 qwen_vl HF"
)
# <---- param settings ---->

cd $PROJECT_DIR
source $CONDA_ACTIVATE_PATH

run_model_bench() {
    local model_name="$1"
    local bench_name="$2"
    local nproc_per_node="$3"
    local conda_env="$4"
    local model_type="$5"
    local inference_file="${PROJECT_DIR}/eval_results/${model_name}/${bench_name}/processed/${model_name}_${bench_name}.jsonl"

    echo "---------------------------------------------"
    echo "Processing Model: $model_name"
    echo "Bench: $bench_name"
    echo "Inference File: $inference_file"
    echo "Using Conda Env: $conda_env"
    echo "---------------------------------------------"

    # activate the corresponding conda environment
    conda activate "$conda_env"
    conda env list

    # check if inference file exists
    if [ -f "$inference_file" ]; then
        echo "Inference file exists for Model: $model_name, Bench: $bench_name. Skipping inference."
    else
        echo "Running inference for Model: $model_name, Bench: $bench_name."

        MODEL_NAME="$model_name" \
        BENCH_NAME="$bench_name" \
        NPROC_PER_NODE="$nproc_per_node" \
        MODEL_TYPE="$model_type" \
        ENV_NAME="$conda_env" \
        PROJECT_DIR="$PROJECT_DIR" \
        zsh ./eval_mmifeval/sh_scripts/run_inference.sh

        # check if inference script runs successfully
        if [ $? -ne 0 ]; then
            echo "Error running inference for Model: $model_name, Bench: $bench_name."
            return 1
        fi
    fi

    # check if inference file exists again
    if [ ! -f "$inference_file" ]; then
        echo "Inference file does not exist for Model: $model_name, Bench: $bench_name."
        return 1
    fi

    # run scoring script
    echo "Running scoring for Model: $model_name, Bench: $bench_name."
    
    MODEL_NAME="$model_name" \
    BENCH_NAME="$bench_name" \
    INFERENCE_FILE="$inference_file" \
    ENV_NAME="$conda_env" \
    PROJECT_DIR="$PROJECT_DIR" \
    zsh ./eval_mmifeval/sh_scripts/score_mp.sh

    # check if scoring script runs successfully
    if [ $? -ne 0 ]; then
        echo "Error running scoring for Model: $model_name, Bench: $bench_name."
        return 1
    fi

    echo "Completed processing for Model: $model_name, Bench: $bench_name."
    echo
}

# iterate over each combination and run function
for pair in "${model_bench_pairs[@]}"; do
    # split model, bench, and environment using space
    model_name=$(echo $pair | awk '{print $1}')
    bench_name=$(echo $pair | awk '{print $2}')
    nproc_per_node=$(echo $pair | awk '{print $3}')
    conda_env=$(echo $pair | awk '{print $4}')  # Get the conda environment
    model_type=$(echo $pair | awk '{print $5}')  # Get the model type

    # call function
    run_model_bench "$model_name" "$bench_name" "$nproc_per_node" "$conda_env" "$model_type"

    # check if function runs successfully
    if [ $? -ne 0 ]; then
        echo "Error processing Model: $model_name, Bench: $bench_name. Exiting."
        exit 1
    fi
done

echo "All model and bench combinations have been processed successfully."
