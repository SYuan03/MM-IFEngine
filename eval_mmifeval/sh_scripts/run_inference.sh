# !/bin/zsh
MODEL_NAME=${MODEL_NAME}
BENCH_NAME=${BENCH_NAME}
CURRENT_TIME=${CURRENT_TIME}
NPROC_PER_NODE=${NPROC_PER_NODE}
PROJECT_DIR=${PROJECT_DIR}

cd $PROJECT_DIR

# set OPENAI_API_KEY and OPENAI_API_BASE
# export OPENAI_API_KEY=
# export OPENAI_API_BASE=

# if CURRENT_TIME is not empty, use CURRENT_TIME, otherwise use current time
if [ -n "$CURRENT_TIME" ]; then
    current_time=$CURRENT_TIME
else
    current_time=$(date +%Y%m%d_%H%M%S)
fi

OUTPUT_DIR_EVAL=${PROJECT_DIR}/logs/eval/${MODEL_NAME}/${BENCH_NAME}
LOG_FILE_EVAL=${OUTPUT_DIR_EVAL}/${current_time}.log
mkdir -p $(dirname "${LOG_FILE_EVAL}")

if [ "$MODEL_TYPE" = "API" ]; then
    PYTHONPATH=$PROJECT_DIR python ./eval_mmifeval/inference_multi_constraint_mp_api.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --current_time $current_time \
    --num_threads $NPROC_PER_NODE \
    --project_dir $PROJECT_DIR \
    2>&1 | tee -a "${LOG_FILE_EVAL}"
else
    PYTHONPATH=$PROJECT_DIR torchrun --nproc_per_node=${NPROC_PER_NODE} ./eval_mmifeval/inference_multi_constraint_mp.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --current_time $current_time \
    --project_dir $PROJECT_DIR \
    2>&1 | tee -a "${LOG_FILE_EVAL}"
fi

