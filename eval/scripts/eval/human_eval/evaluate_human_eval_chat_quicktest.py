import os
import re
import time
import openai
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset
from human_eval.data import write_jsonl, read_problems

import sys
sys.path.append("./scripts/eval")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.0)
args = parser.parse_args()
problems = read_problems()
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv_hf(example, tokenizer):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
                f"```python\n"
                f"{example['prompt']}\n"
                f"```\n"
            )
    system_prompt = open("scripts/eval/system_prompt.md").read()

    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
    ]

    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

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
        enforce_eager=True,
    )
    sampling_params = SamplingParams(max_tokens=4096,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=["<|eot_id|>"],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)

    completions = [match_code(output.outputs[0].text) for output in outputs]
    return completions

def make_signature(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    return signature

samples = []
problems = Dataset.from_pandas(pd.DataFrame(problems).T)
problems = problems.map(lambda x: {"signature": make_signature(x)}, cache_file_name="cache/human_eval", load_from_cache_file=False)
problems = problems.map(lambda x: {"instruction": make_conv_hf(x, tokenizer)}, cache_file_name="cache/human_eval", load_from_cache_file=False)

completions = generate_sample_batch(problems["instruction"])
problems = problems.add_column("completion", completions)
problems = problems.map(lambda x: {"completion": x["prompt"] + x["completion"]})
samples = problems.to_pandas().to_dict(orient="records")

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(os.path.join(args.save_dir)):
    os.mkdir(os.path.join(args.save_dir))
write_jsonl(output_filepath, samples)
