import os
from time import sleep

try:
    import google.generativeai as genai
    from google.generativeai import GenerationConfig
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class GeminiRunner(BaseRunner):
    client = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client = genai.GenerativeModel(model.model_name)
        self.generation_config = GenerationConfig(
            candidate_count=1,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    def _run_single(self, prompt: str) -> list[str]:

        def __run_single(counter):
            try:
                return self.client.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=GeminiRunner.safety_settings,
                )
            except Exception as e:
                print("Exception: ", repr(e), "Sleeping for 20 seconds...")
                sleep(20 * (11 - counter))
                counter = counter - 1
                if counter == 0:
                    print(f"Failed to run model for {prompt}!")
                    print("Exception: ", repr(e))
                    raise e
                return __run_single(counter)

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e

        new_outputs = []
        for output in outputs:
            try:
                new_outputs.append(output.text)
            except Exception as e:
                print("Cannot extract text exception: ", repr(e))
                print(output.__dict__)
                new_outputs.append("")
        outputs = new_outputs

        return outputs
