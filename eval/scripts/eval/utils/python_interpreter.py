from typing import Mapping
import re
import signal
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from typing import Any
from IPython import get_ipython
import subprocess
from tqdm import tqdm
# class PythonREPL:
#     """A tool for running python code in a REPL."""

#     name = "PythonREPL"
#     # This PythonREPL is not used by the environment; It is THE ENVIRONMENT.
#     signature = "NOT_USED"
#     description = "NOT_USED"

#     def __init__(
#         self,
#         user_ns: Mapping[str, Any] = {},
#         timeout: int = 3,
#     ) -> None:
#         super().__init__()
#         self.user_ns = user_ns
#         self.timeout = timeout
#         self.reset()

#     @contextmanager
#     def time_limit(self, seconds):
#         def signal_handler(signum, frame):
#             raise TimeoutError(f"Timed out after {seconds} seconds.")

#         signal.signal(signal.SIGALRM, signal_handler)
#         signal.alarm(seconds)
#         try:
#             yield
#         finally:
#             signal.alarm(0)  # Disable the alarm

#     def reset(self) -> None:
#         InteractiveShell.clear_instance()
#         self.shell = InteractiveShell.instance(
#             # NOTE: shallow copy is needed to avoid
#             # shell modifying the original user_ns dict
#             user_ns=dict(self.user_ns),
#             colors="NoColor",
#         )

#     def __call__(self, query: str) -> str:
#         """Use the tool and return observation"""
#         with self.time_limit(self.timeout):
#             # NOTE: The timeout error will be caught by the InteractiveShell

#             # Capture all output
#             with io.capture_output() as captured:
#                 _ = self.shell.run_cell(query, store_history=True)
#             output = captured.stdout

#             if output == "":
#                 output = "[Executed Successfully with No Output]"

#             # replace potentially sensitive filepath
#             # e.g., File /mint/mint/tools/python_tool.py:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
#             # with File <filepath>:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
#             # use re
#             output = re.sub(
#                 # r"File (/mint/)mint/tools/python_tool.py:(\d+)",
#                 r"File (.*)python_tool.py:(\d+)",
#                 r"File <hidden_filepath>:\1",
#                 output,
#             )
#             if len(output) > 1000:
#                 output = output[:1000] + "...\n[Output Truncated]"

#         return output

import os
class PythonREPL():
    def __init__(self, timeout=5, tmp_file="cache/tmp"):
        self.timeout = timeout
        self.tmp_file = tmp_file
        os.system(f"touch {self.tmp_file}.py" )

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm
 
    def __call__(self, query: str) -> str:
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)

        with open(f'{self.tmp_file}.py', "w") as f:
            f.write(query)
        
        with self.time_limit(self.timeout):
            result = subprocess.run(
                    ['python3', f'{self.tmp_file}.py'], capture_output=True, check=False, text=True, timeout=self.timeout)

            if result.returncode == 0:
                output = result.stdout
                return True, output.strip()
            else:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif self.tmp_file in m:
                        st = m.index('"/') + 1 if '"/' in m else 0
                        ed = m.index(f'/{self.tmp_file}.py') + 1 if f'/{self.tmp_file}.py' in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()
        
    
def postprocess_completion(executor, completion):
    # solution = re.findall(r"<solution>(.*?)</solution>", completion, re.DOTALL)
    # assert len(solution) <= 1
    # has_solution = bool(len(solution) > 0)
    
    executions = ["!" + code for code in re.findall(r"```bash(.*?)```", completion, re.DOTALL) if "!" not in code]
    executions.extend(re.findall(r"```python(.*?)```", completion, re.DOTALL))
    
    if len(executions) == 0: # directly return cot result
        return completion
    else:

        ### Jupyter
        # # put code into notebook
        # execution_outputs = [executor(code) for code in executions]     
        # # extract answer
        # extracted_outputs = [(int(match.group(1)), match.group(2)) for match in (re.match(r'Out\[(\d+)\]: (.+)', s) for s in execution_outputs) if match]
        
        ### Python
        execution_outputs = []
        for code in executions:
            try: 
                success, output = executor(code)
            except TimeoutError:
                print("time out")
                # success = False
                output = ""
            else:
                output = output if success else ""
            execution_outputs.append(output)
        extracted_outputs = execution_outputs

        # if len(extracted_outputs) < 1:
        #     return ""

        for index in range(1, len(extracted_outputs) + 1):
            # if len(extracted_outputs[-index]) < 2:
            #     return None
            # extracted_solution = str(extracted_outputs[-index][1]).strip() # Jupyter
            extracted_solution = str(extracted_outputs[-index]).strip() # Python
            break

        return extracted_solution


def postprocess_completions(completion_list):
    executor = PythonREPL()
    
    solution_list = []
    # for completion in tqdm(completion_list, total=len(completion_list), desc="Executing"):
    for completion in completion_list:
        # executor.reset()
        solution_list.append(postprocess_completion(executor, completion))
    # executor.reset()
    del executor
    # ipython = get_ipython()
    # if ipython:
    #     ipython.exit()
    return solution_list



if __name__ == "__main__":
    code = """
Step 1: First, let's calculate the total number of eggs laid by Janet's ducks in a day.
Step 2: Next, let's calculate the number of eggs Janet eats for breakfast each day.
Step 3: Then, let's calculate the number of eggs Janet bakes for her friends each day.
Step 4: Finally, let's calculate the number of eggs Janet sells at the farmers' market each day.
Step 5: To find the total amount of money Janet makes each day at the farmers' market, we can multiply the number of eggs she sells by the price per egg.
```python
# Step 6: Calculate the total number of eggs laid by Janet's ducks in a day.
total_eggs_per_day = 16
# Step 7: Calculate the number of eggs Janet eats for breakfast each day.
eggs_eaten_per_day = 3
# Step 8: Calculate the number of eggs Janet bakes for her friends each day.
eggs_baked_per_day = 4
# Step 9: Calculate the number of eggs Janet sells at the farmers' market each day.
eggs_sold_per_day = total_eggs_per_day - eggs_eaten_per_day - eggs_baked_per_day
# Step 10: Calculate the total amount of money Janet makes each day at the farmers' market.
price_per_egg = 2
total_money_per_day = eggs_sold_per_day * price_per_egg
total_money_per_day
```
Answer:
12

"""
    postprocess_completion(code)
