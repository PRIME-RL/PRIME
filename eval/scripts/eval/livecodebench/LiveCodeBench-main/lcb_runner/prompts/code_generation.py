import json

try:
    from anthropic import HUMAN_PROMPT, AI_PROMPT
except ImportError:
    HUMAN_PROMPT = None
    AI_PROMPT = None

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


class PromptConstants:
    SYSTEM_MESSAGE_o1 = open("../../../../scripts/eval/system_prompt.md").read()

    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program."

    SYSTEM_MESSAGE_GEMINI = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Do NOT use system calls like `exit` in the generated program."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example: 
```python 
# YOUR CODE HERE
```"""

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


def get_generic_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_cllama_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"[PYTHON]\n{question.starter_code}\n[/PYTHON]\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"[PYTHON]\n# WRITE YOUR CODE HERE\n[/PYTHON]\n\n"
    prompt += f"### ANSWER (use the provided delimiters, read the inputs from stdin and write response to stdout)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Instruction: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += (
            f"### Instruction: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        )
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Response:\n\n"
    return prompt


def get_qwen_question_template_answer(question: CodeGenerationProblem):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/abacus/models/Qwen1.5-72B-Chat/", padding_side="left", use_fast=False
    )
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"

    messages = [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt


def get_magicoder_question_template_answer(question: CodeGenerationProblem):
    prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"@@ Response\n"
    return prompt


def get_wizard_question_template_answer(question: CodeGenerationProblem):
    prompt = f"""### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example:
```python 
# YOUR CODE HERE
```
"""
    prompt += f"{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Response:\n\n"
    return prompt


def get_phind_question_template_answer(question: CodeGenerationProblem):
    prompt = f"{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"\n\n### Assistant"
    return prompt


def get_codeqwen_question_template_answer(question: CodeGenerationProblem):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt


with open("lcb_runner/prompts/few_shot_examples/generation/func.json") as f:
    func = json.load(f)

with open("lcb_runner/prompts/few_shot_examples/generation/stdin.json") as f:
    stdin = json.load(f)


def get_base_model_question_template_answer(question: CodeGenerationProblem):
    if question.starter_code:
        examples_json = func
    else:
        examples_json = stdin

    def get_example_prompt(example):
        prompt = ""
        prompt += "### Question\n"
        prompt += example["question"]
        prompt += "\n\n"
        if question.starter_code:
            prompt += "### Starter Code\n"
            prompt += example["sample_code"]
            prompt += "\n\n"
        prompt += "### Answer\n\n"
        prompt += example["answer"]
        if example["answer"]:
            prompt += "\n\n"
        return prompt

    prompt = ""
    prompt += get_example_prompt(examples_json[0])
    prompt += get_example_prompt(
        {
            "question": question.question_content,
            "sample_code": question.starter_code,
            "answer": "",
        }
    )
    return prompt


def format_prompt_generation(
    question: CodeGenerationProblem, LanguageModelStyle: LMStyle
) -> str:
    if LanguageModelStyle in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI]:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.OpenAIReason:
        chat_messages = [
            {
                "role": "user",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC
                + "\n\n"
                + get_generic_question_template_answer(question),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.LLaMa3:
        # SYSTEM_MESSAGE_o1
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_o1,
                # "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question) + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.",
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "/home/test/testdata/models/Meta-Llama-3.1-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    if LanguageModelStyle == LMStyle.Qwen25Math:
        # SYSTEM_MESSAGE_o1
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_o1,
                # "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question) + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.",
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "/home/test/test05/whb/models/Qwen2.5-Math-7B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )

    if LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n"
        prompt += f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n"
        prompt += f"{get_generic_question_template_answer(question).rstrip()}\n"
        prompt += f"{AI_PROMPT}"
        return prompt

    if LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question).rstrip(),
            }
        ]
        return system, prompt

    if LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GEMINI}\n"
        prompt += f"{get_generic_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n"
        prompt += f"{get_generic_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
            {
                "role": "user",
                "content": get_generic_question_template_answer(question),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.CohereCommand:
        chat_messages = [
            {
                "role": "System",
                "message": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        message = get_generic_question_template_answer(question)
        return chat_messages, message

    if LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n"
        prompt += f"{get_deepseekcode_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN}\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n"
        prompt += f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n"
        prompt += f"<</SYS>>\n\n"
        prompt += f"{get_cllama_question_template_answer(question)}\n"
        prompt += f"[/INST]"
        return prompt

    if LanguageModelStyle == LMStyle.MagiCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_MAGIC}\n"
        prompt += f"{get_magicoder_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.WizardCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_WIZARD}\n\n"
        prompt += f"{get_wizard_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.Phind:
        prompt = f"### System Prompt\n\n"
        prompt += f"{PromptConstants.SYSTEM_MESSAGE_PHIND}\n\n"
        prompt += f"### User Message\n\n"
        prompt += f"{get_phind_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.OC:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n"
        prompt += f"{get_generic_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.Eurusx:
        prompt = "[INST] Write Python code to solve the task:\n"
        prompt += f"{get_generic_question_template_answer(question)}"
        prompt += "[/INST]"
        return prompt

    if (
        LanguageModelStyle == LMStyle.Smaug2
        or LanguageModelStyle == LMStyle.Qwen1point5
    ):
        prompt = f"{get_qwen_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.GenericBase:
        prompt = get_base_model_question_template_answer(question)
        return prompt

    if LanguageModelStyle == LMStyle.DracarysQwen:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN}\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.DracarysLlama:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question),
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "abacusai/Dracarys-Llama-3.1-70B-Instruct",
            padding_side="right",
            use_fast=False,
        )

        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )

    raise NotImplementedError(
        f"LanguageModelStyle {LanguageModelStyle} not implemented"
    )


def test():
    import pathlib

    base_dir = "logs/example_prompts/generation"
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    for lmstyle in LMStyle:
        generation_problem = CodeGenerationProblem(
            "title",
            "question-content",
            "leetcode",
            "question_id",
            "contest_id",
            "contest_date",
            "",
            "easy",
            "[]",
            "[]",
            "{}",
        )
        prompt1 = format_prompt_generation(generation_problem, lmstyle)
        with open(f"{base_dir}/{lmstyle}_1.txt", "w") as f:
            try:
                f.write(prompt1)
            except TypeError:
                f.write(json.dumps(prompt1))

        generation_problem.starter_code = "starter code"
        prompt2 = format_prompt_generation(generation_problem, lmstyle)
        with open(f"{base_dir}/{lmstyle}_2.txt", "w") as f:
            try:
                f.write(prompt2)
            except TypeError:
                f.write(json.dumps(prompt2))


if __name__ == "__main__":
    test()
