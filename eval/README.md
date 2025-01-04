# Evaluation

### Requirements

Before the evaluation, please install the required packages with the following command:

We use different virtual environments for different test sets. The reason for this is that the version of the package is critical when evaluating the math and code benchmarks, and we build virtual environments from the official repositories of [Eurus](https://github.com/OpenBMB/Eurus), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), respectively.

For AIME 2024, AMC, MATH-500, HumanEval, MBPP, and LeetCode, we use the following virtual environment.

```bash
git clone https://github.com/PRIME-RL/PRIME.git
cd eval
conda create -n prime python==3.10
conda activate prime
pip install -r requirements_prime.txt
```

For Minerva Math and OlympiadBench, we use the following virtual environment.

```bash
# git clone https://github.com/PRIME-RL/PRIME.git
# cd eval
conda create -n qwen_math python==3.10
conda activate qwen_math
pip install -r requirements_qwen_math.txt
```

For LiveCodeBench, we use the following virtual environment.

```bash
# git clone https://github.com/PRIME-RL/PRIME.git
# cd eval
conda create -n lcb python==3.10
conda activate lcb
pip install -r requirements_lcb.txt
```

### Eval

1. Set your `conda.sh` path [here]() so that different environments can be activated during testing. For example, I set to `/path/anaconda3/etc/profile.d/conda.sh`

2. Set the output directory [here]() to save the test results.

3. Download the model, you can download the model locally in [HuggingFace](https://huggingface.co/PRIME-RL).

4. Evaluate models with the following command (Make sure you have `cd eval`):

   ```bash
   bash scripts/eval/run.sh
   # You can specify the data set to be tested in the my_array array of the script
   ```

   



