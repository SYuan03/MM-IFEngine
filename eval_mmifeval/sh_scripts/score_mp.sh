#!/bin/zsh
MODEL_NAME=${MODEL_NAME}
BENCH_NAME=${BENCH_NAME}
CURRENT_TIME=${CURRENT_TIME}
PROJECT_DIR=${PROJECT_DIR}

cd $PROJECT_DIR

# if CURRENT_TIME is not empty, use CURRENT_TIME, otherwise use current time
if [ -n "$CURRENT_TIME" ]; then
    current_time=$CURRENT_TIME
else
    current_time=$(date +%Y%m%d_%H%M%S)
fi

OUTPUT_DIR_SCORE=${PROJECT_DIR}/logs/eval/${MODEL_NAME}/${BENCH_NAME}
LOG_FILE_SCORE=${OUTPUT_DIR_SCORE}/${current_time}.log
mkdir -p $(dirname "${LOG_FILE_SCORE}")


if [ -n "$INFERENCE_FILE" ]; then
    PYTHONPATH=$PROJECT_DIR python ./eval_mmifeval/score_multi_constraint_mp_with_image_v11.py \
    --inference_file $INFERENCE_FILE \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --project_dir $PROJECT_DIR \
    2>&1 | tee -a "${LOG_FILE_SCORE}"
else
    PYTHONPATH=$PROJECT_DIR python ./eval_mmifeval/score_multi_constraint_mp_with_image_v11.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --project_dir $PROJECT_DIR \
    2>&1 | tee -a "${LOG_FILE_SCORE}"
fi