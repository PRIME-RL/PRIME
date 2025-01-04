#!/bin/bash
source xx/anaconda3/etc/profile.d/conda.sh   # set to your path
MODEL_CKPT=$1
MODEL_NAME=$(basename "$MODEL_CKPT")
OUTPUT_DIR="results/$MODEL_NAME" # output dir
mkdir -p $OUTPUT_DIR


# Specify the test data set
my_array=(humaneval mbpp leetcode math500 amc aime qwen livecodebench)
# my_array=(livecodebench)



if [[ " ${my_array[@]} " =~ " humaneval " ]]; then
    conda activate prime
    echo "running humaneval"
    mkdir -p $OUTPUT_DIR/human_eval_chat
    touch scripts/eval/human_eval/human_eval
    python3 scripts/eval/human_eval/evaluate_human_eval_chat_quicktest.py \
        --model $MODEL_CKPT \
        --save_dir $OUTPUT_DIR/human_eval_chat

    python ./eval-harness/human-eval/human_eval/evaluate_functional_correctness.py --sample_file $OUTPUT_DIR/human_eval_chat/samples.jsonl

    nohup python ./eval-harness/human-eval/human_eval/evaluate_functional_correctness.py \
            --sample_file $OUTPUT_DIR/human_eval_chat/samples.jsonl \
            >$OUTPUT_DIR/human_eval_chat/result.txt 2>&1 &
fi

if [[ " ${my_array[@]} " =~ " mbpp " ]]; then
    conda activate prime
    echo "running mbpp"
    mkdir -p $OUTPUT_DIR/mbpp_chat
    touch cache/mbpp
    python3 -u scripts/eval/mbpp/run_mbpp_chat_quicktest.py \
        --model $MODEL_CKPT \
        --input_data data/mbpp/new_mbpp.json \
        --save_dir $OUTPUT_DIR/mbpp_chat
fi


if [[ " ${my_array[@]} " =~ " leetcode " ]]; then
    conda activate prime
    echo "running leetcode"
    mkdir -p $OUTPUT_DIR/leetcode_chat
    touch cache/leetcode
    python3 scripts/eval/leetcode/evaluate_leetcode_chat_quicktest.py \
        --model $MODEL_CKPT \
        --input_data data/leetcode/leetcode-test.json \
        --save_dir $OUTPUT_DIR/leetcode_chat

    python ./scripts/eval/leetcode/evaluate_leetcode.py --generation_path $OUTPUT_DIR/leetcode_chat/samples.jsonl --temp_dir ./cache

    nohup python ./scripts/eval/leetcode/evaluate_leetcode.py \
            --generation_path $OUTPUT_DIR/leetcode_chat/samples.jsonl \
            --temp_dir ./cache \
            >$OUTPUT_DIR/leetcode_chat/result.txt 2>&1 &
fi


if [[ " ${my_array[@]} " =~ " amc " ]]; then
    conda activate prime
    echo "running amc_chat(numina)"
    mkdir -p $OUTPUT_DIR/amc_chat
    python3 -u scripts/eval/amc/evaluate_amc_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir  data/AI-MO/aimo-validation-amc \
        --save_dir $OUTPUT_DIR/amc_chat
fi

if [[ " ${my_array[@]} " =~ " aime " ]]; then
    conda activate prime
    # AIME2024 chat
    echo "running aime_chat(numina)"
    mkdir -p $OUTPUT_DIR/aime_chat
    python3 -u scripts/eval/aime/evaluate_aime_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir  data/AI-MO/aimo-validation-aime \
        --save_dir $OUTPUT_DIR/aime_chat
fi


if [[ " ${my_array[@]} " =~ " math500 " ]]; then
    conda activate prime
    # math chat
    echo "running math_chat 500"
    mkdir -p $OUTPUT_DIR/math_chat
    python3 -u scripts/eval/math/evaluate_math_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir data/math500 \
        --save_dir $OUTPUT_DIR/math_chat 
fi

if [[ " ${my_array[@]} " =~ " qwen " ]]; then
    conda activate qwen_math
    echo "running qwen math eval datasets"
    cd ./scripts/eval/Qwen25-Math/evaluation
    PROMPT_TYPE="qwen25-math-cot"
    MODEL_NAME_OR_PATH=$MODEL_CKPT
    mkdir -p $OUTPUT_DIR/qwen_math
    bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR/qwen_math
    cd ../../../../
fi


if [[ " ${my_array[@]} " =~ " livecodebench " ]]; then
    conda activate lcb
    echo "running livecodebench"
    cd ./scripts/eval/livecodebench/LiveCodeBench-main
    mkdir -p $OUTPUT_DIR/livecodebench
    python -m lcb_runner.runner.main --model $MODEL_CKPT --scenario codegeneration --evaluate --release_version release_v4 --output_path $OUTPUT_DIR/livecodebench
    # v2
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-05-31 >$OUTPUT_DIR/livecodebench/lcb_v2.txt 2>&1 &
    # v3
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-08-03 >$OUTPUT_DIR/livecodebench/lcb_v3.txt 2>&1 &
    # v4
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_v4.txt 2>&1 &
    # 08-
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2024-08-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_08_11.txt 2>&1 &
    cd ../../../../
fi








