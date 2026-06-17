import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from  src.models import FunctionDefinition, TestPrompt
from llm_sdk import Small_LLM_Model


class VocabularyManager(BaseModel):
    path: str
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    def model_post_init(self, __context: object) -> None:
        try:
            with open(self.path, "r") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"[VocabularyManager] ERROR: vocab file not found at '{self.path}'")
            return
        except json.JSONDecodeError as e:
            print(f"[VocabularyManager] ERROR: invalid JSON in '{self.path}': {e}")
            return

        # Adjust this block based on what your inspection showed.
        # Assumes raw_data is {token_string: token_id}
        self.token_to_id = raw_data
        self.id_to_token = {
            token_id: token_str for token_str, token_id in raw_data.items()
        }

    def get_tokens_starting_with(self, prefix: str) -> list[int]:
        result: list[int] = []
        for token_str, token_id in self.token_to_id.items():
            if token_str.startswith(prefix):
                result.append(token_id)
        return result


def greedy_next_token(logits: np.ndarray, valid_ids: list[int]) -> int:
    masked = logits.copy()
    for i in range(len(masked)):
        if i not in valid_ids:
            masked[i] = -np.inf
    return int(np.argmax(masked))
def select_function_name(prompt:str,functions:list[FunctionDefinition],model:Small_LLM_Model,vocab:VocabularyManager) -> str:
    candidates = [fonction.name for fonction in functions]
    prompt_context = (
        "You are a function calling assistant.\n"
        + "\n".join(
            f"{function.name}: {function.description}"
            for function in functions
        )
        + f"\nUser request: {prompt}\n"
        + '{"name": "'
    )
    input_ids = model.encode(prompt_context)[0].tolist()
    generated_so_far = ""
    current_ids = input_ids
    step = 0
    max_steps = 50

    while True:
        step += 1

        if step > max_steps:
            raise ValueError("Generation exceeded max steps")

    logits = model.get_logits_from_input_ids(current_ids)
    logits_np = np.array(logits)
    valid_ids = []
    for token_str, token_id in vocab.token_to_id.items():
        for name in candidates:
            if name.startswith(generated_so_far + token_str):
                valid_ids.append(token_id)
                break
    if generated_so_far in candidates:
            return generated_so_far