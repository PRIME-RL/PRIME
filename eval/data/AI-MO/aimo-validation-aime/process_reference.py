import json
from transformers import AutoTokenizer
from tqdm import tqdm


def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

path = "/home/test/test05/whb/data/test_data_o1/AI-MO/aimo-validation-amc/aimo-validation-amc.jsonl"
data = read_jsonl_file(path)
new_data = []
for line in data:
    line["answer"] = str(line["answer"])
    new_data.append(line)
write_jsonl_file(path, new_data)
