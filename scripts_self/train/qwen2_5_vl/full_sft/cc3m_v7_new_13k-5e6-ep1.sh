#!/bin/zsh
ENV_NAME=llamafactory_qwen_2_5
SCRIPT_NAME=cc3m_v7_new_13k-5e6-ep1
HOME_DIR=

source ${HOME_DIR}/.zshrc

echo "HF_HOME: $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"

export WANDB_API_KEY=3cd9087c559cfb334ec9ea75bd4a5c5bfc287556

export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""

export HF_HOME=/fs-computility/mllm1/shared/dsy_volcengine/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# activate the environment
source ${HOME_DIR}/miniconda3/bin/activate
conda activate $ENV_NAME
conda env list

OUTPUT_DIR=saves/qwen2_5_vl-7b/full_sft/${SCRIPT_NAME}
LOG_DIR=${OUTPUT_DIR}/logs

mkdir -p $LOG_DIR

# run the training
llamafactory-cli train examples/qwen2_5_vl/full_sft/${SCRIPT_NAME}.yaml \
2>&1 | tee -a "${LOG_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt"