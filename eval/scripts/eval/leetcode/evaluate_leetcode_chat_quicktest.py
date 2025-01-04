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
import json
from transformers import AutoTokenizer
import sys
sys.path.append("./scripts/eval")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--input_data", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.)
args = parser.parse_args()

problems = pd.read_json(args.input_data, lines=True)
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

import re
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
        gpu_memory_utilization=0.80,
    )
    sampling_params = SamplingParams(max_tokens=8192,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=["<|eot_id|>"],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = ["```python\n" +  match_code(output.outputs[0].text) + "\n```" for output in outputs]
    outputs = [output.outputs[0].text for output in outputs]
    return completions,outputs


def make_conv_hf(example, tokenizer):
    system_prompt = open("scripts/eval/system_prompt.md").read()
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["prompt_sft"] + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat 

samples = []
del problems["start_time"]

tokenizer = AutoTokenizer.from_pretrained(args.model)
problems["instruction"] = problems.apply(lambda row: make_conv_hf(row, tokenizer), axis=1)


completions,outputs = generate_sample_batch(problems["instruction"])
for i in range(len(completions)):
    if "class Solution:" not in completions[i]:
        completions[i] = completions[i].replace("```python", "").replace("```", "")
        completions[i] = "\n".join(["    " + line for line in completions[i].split("\n")])
        completions[i] = "class Solution:\n" + completions[i]
        completions[i] = "```python\n" + completions[i] + "\n```"
    
problems["output"] = completions
problems["raw_output"] = outputs
samples = problems.to_dict(orient="records")
output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)
