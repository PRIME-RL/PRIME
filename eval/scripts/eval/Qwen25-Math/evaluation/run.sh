PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="/home/test/testdata/models/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR=""
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR