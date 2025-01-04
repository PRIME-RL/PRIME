import json
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import os
import torch
import openai
import argparse
from vllm import LLM, SamplingParams
import time
import re

import sys
sys.path.append("./scripts/eval")

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

STOP_WORDS = []
def match_code(s):
    pattern = r'```python(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if 'def ' in code_block:
                return code_block
        return sol[-1]
    
    pattern = r'```(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if 'def ' in code_block:
                return code_block
        return sol[-1]
    
    return s.split('```')[0]

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(max_tokens=4096,
                                    temperature=0.0,
                                    n=1,
                                    stop=STOP_WORDS)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    raw_completions = [output.outputs[0].text for output in outputs]
    completions = [match_code(output.outputs[0].text) for output in outputs]
    return completions,raw_completions


def make_signature(code):
    signature = [line for line in code.split("\n") if line.strip().startswith("def ")][0]
    signature = signature.lstrip("def ").replace(" ", "").rstrip(":").strip().replace(",", ", ")
    assert ":" not in signature
    return signature


from transformers import AutoTokenizer
def make_conv_hf(signature, description, test_list, tokenizer):
    description = description.split(" https://www.")[0]
    testcase = test_list[0]
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python and ```\n"
                f"{description}\n"
                f"The code should pass the following test cases:>>> {testcase}\n\n Let's coding step by step.\n"
            )
    system_prompt = open("scripts/eval/system_prompt.md").read()
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
    ]

    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

import contextlib
import signal
class TimeoutException(Exception):
    pass
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def exec_helper(code):
    with time_limit(3):
        exec(compile(code, filename="mbpp", mode='exec'), globals())

def evaluate(dataset):
    correct = 0
    format_error = 0
    exec_error = 0

    for example in dataset.to_dict(orient="records"):
        completion = example["completion"]
        # remove texts
        code = completion.split("\n")
        code_ = []
        for c in code:
            if len(c.lstrip()) == len(c) and not c.startswith("def"):
                continue
            code_.append(c)
        code = "\n".join(code_)

        function = code
        test_cases = "\n".join(example["test_list"]).replace("\/", "/")
        test_run = "\n".join([
            function,
            test_cases,
        ])

        # define function
        try:
            exec_helper(function)
        except Exception as e:
            print(function)
            print("Error",e)
            format_error += 1
            continue           

        try:
            # run test case
            exec_helper(test_cases)
            exec_helper(test_run)
        except:
            exec_error += 1
            continue
        else:
            correct += 1
    print("correct: ", correct)
    print("exec_error: ", exec_error)
    print("format_error: ", format_error)
    return 100 * (correct / len(dataset)), 100 * (exec_error / len(dataset)), 100 * (format_error / len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--input_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = pd.read_json(args.input_data, lines=False)
    dataset["signature"] = dataset.apply(lambda row: make_signature(row["code"]), axis=1)
    for signature in dataset["signature"]:
        STOP_WORDS.append("\n\nprint(" + signature.split("(")[0].strip())
    dataset["prompt"] = dataset.apply(lambda row: make_conv_hf(row["signature"], row["prompt"], row["test_list"], tokenizer), axis=1)
    completions,raw_completions = generate_sample_batch(dataset["prompt"].tolist())
    dataset["raw_completion"] = raw_completions
    dataset["completion"] = completions
    del dataset["source_file"]
    dataset["completion"] = dataset.apply(lambda row: "def" + row["completion"] if "def" not in row["completion"] else row["completion"], axis=1)
    dataset.to_json(os.path.join(args.save_dir, "mbpp_completion.json"))
    accuracy, exec_error, format_error = evaluate(dataset)
    
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error})
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error}, file=f)
