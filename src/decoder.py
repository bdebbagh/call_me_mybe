import json

import numpy as np
from pydantic import BaseModel
from src.models import FunctionDefinition
from llm_sdk import Small_LLM_Model


class VocabularyManager(BaseModel):
    """Loads and manages the model vocabulary for token lookups."""

    path: str
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    def model_post_init(self, __context: object) -> None:
        """Runs automatically after object is created. Loads vocab.json."""
        try:
            with open(self.path, "r") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"[VocabularyManager] ERROR: vocab file not found at '{self.path}'")
            return
        except json.JSONDecodeError as e:
            print(f"[VocabularyManager] ERROR: invalid JSON in '{self.path}': {e}")
            return

        self.token_to_id = raw_data
        self.id_to_token = {
            token_id: token_str for token_str, token_id in raw_data.items()
        }

    def get_tokens_starting_with(self, prefix: str) -> list[int]:
        """Returns list of token IDs whose text starts with prefix."""
        result: list[int] = []
        for token_str, token_id in self.token_to_id.items():
            if token_str.startswith(prefix):
                result.append(token_id)
        return result


def greedy_next_token(logits: np.ndarray, valid_ids: list[int]) -> int:
    """Masks all invalid token IDs to -inf. Returns best valid token ID."""
    masked = logits.copy()
    for i in range(len(masked)):
        if i not in valid_ids:
            masked[i] = -np.inf
    return int(np.argmax(masked))


def select_function_name(
    prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: VocabularyManager,
) -> str:
    """Uses constrained decoding to pick which function matches the prompt.

    Returns the exact function name string e.g. 'fn_add_numbers'.
    """

    candidates = [fn.name for fn in functions]

    prompt_context = (
        "You are a function calling assistant.\n"
        + "\n".join(
            f"{fn.name}: {fn.description}"
            for fn in functions
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
            raise ValueError("select_function_name exceeded max steps")

        logits = model.get_logits_from_input_ids(current_ids)
        logits_np = np.array(logits)

        if generated_so_far in candidates:
            return generated_so_far

        valid_ids: list[int] = []
        for token_str, token_id in vocab.token_to_id.items():
            for name in candidates:
                if name.startswith(generated_so_far + token_str):
                    valid_ids.append(token_id)
                    break

        next_token_id = greedy_next_token(logits_np, valid_ids)
        current_ids.append(next_token_id)
        generated_so_far += vocab.id_to_token[next_token_id]


def _is_valid_number_prefix(s: str) -> bool:
    """Returns True if s is a valid prefix of a number like 2.0 or -3.14."""
    if s == "" or s == "-" or s == ".":
        return True
    try:
        float(s)
        return True
    except ValueError:
        if s.endswith("."):
            try:
                float(s[:-1])
                return True
            except ValueError:
                return False
        return False


def extract_arguments(
    prompt: str,
    function_name: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: VocabularyManager,
) -> dict[str, str | int | float]:
    """Extracts argument values from the prompt for the chosen function.

    Works for any function with any number of parameters.
    Returns dict like {"a": 2.0, "b": 3.0} or {"name": "shrek"}.
    """

    # --- step 1: find the chosen function ---
    chosen_function = None
    for fn in functions:
        if fn.name == function_name:
            chosen_function = fn
            break

    if chosen_function is None:
        raise ValueError(f"Function '{function_name}' not found in definitions")

    # --- step 2: build description of all parameters ---
    # example: "a (number), b (number)"
    # this helps the model understand what each parameter means
    params_description = ""
    for pname, pdef in chosen_function.parameters.items():
        if params_description != "":
            params_description += ", "
        params_description += f"{pname} ({pdef.type})"

    # --- step 3: get list of parameters to extract ---
    param_items = list(chosen_function.parameters.items())

    # will hold extracted values as we go
    extracted: dict[str, str | int | float] = {}

    # --- step 4: loop once per parameter ---
    for param_name, param_def in param_items:

        param_type = param_def.type  # "number" or "string"

        # build the JSON text of parameters already extracted
        already_extracted = ""
        for k, v in extracted.items():
            if already_extracted != "":
                already_extracted += ", "
            already_extracted += f'"{k}": {json.dumps(v)}'

        # build params_so_far — what goes inside "parameters": { ... }
        if already_extracted != "":
            params_so_far = already_extracted + f', "{param_name}": '
        else:
            params_so_far = f'"{param_name}": '

        # build the full prompt
        # includes function description and parameter names
        # so model understands what value to extract
        # example for regex param:
        #   Function: fn_substitute_string_with_regex
        #   Description: Replace all occurrences matching a regex pattern in a string.
        #   Parameters needed: source_string (string), regex (string), replacement (string)
        #   User request: Replace all vowels in 'Programming is fun' with asterisks
        #   {"name": "fn_substitute_string_with_regex", "parameters": {
        #     "source_string": "Programming is fun", "regex": "
        prompt_context = (
            "You are a function calling assistant.\n"
            + f"Function: {function_name}\n"
            + f"Description: {chosen_function.description}\n"
            + f"Parameters needed: {params_description}\n"
            + f"User request: {prompt}\n"
            + '{"name": "' + function_name + '", "parameters": {'
            + params_so_far
        )

        # for strings: add opening quote so model continues from inside the string
        if param_type == "string":
            prompt_context += '"'

        # tokenize the prompt
        input_ids = model.encode(prompt_context)[0].tolist()
        current_ids = input_ids
        generated_value = ""
        step = 0
        max_steps = 100

        # --- generation loop for this one parameter ---
        while True:
            step += 1
            if step > max_steps:
                raise ValueError(
                    f"extract_arguments exceeded max steps for '{param_name}'"
                )

            # get 150,000 scores from model
            logits = model.get_logits_from_input_ids(current_ids)
            logits_np = np.array(logits)

            # NUMBER: only allow digit tokens
            if param_type == "number":

                # get the token the model most wants to generate (no constraints)
                best_free_token = vocab.id_to_token.get(int(np.argmax(logits_np)), "")

                # if model's top choice is not a digit → number is complete
                best_stripped = best_free_token.strip()
                all_digit = True
                for c in best_stripped:
                    if c not in "0123456789.-eE":
                        all_digit = False
                        break

                if not all_digit and generated_value != "":
                    break

                # otherwise find valid digit tokens and pick best
                valid_ids = []
                for token_str, token_id in vocab.token_to_id.items():
                    stripped = token_str.strip()
                    if stripped == "":
                        continue
                    all_valid = True
                    for c in stripped:
                        if c not in "0123456789.-eE":
                            all_valid = False
                            break
                    if all_valid:
                        if _is_valid_number_prefix(generated_value + stripped):
                            valid_ids.append(token_id)

                if not valid_ids:
                    break

                next_token_id = greedy_next_token(logits_np, valid_ids)
                next_token_str = vocab.id_to_token[next_token_id].strip()

                if not _is_valid_number_prefix(generated_value + next_token_str):
                    break

                generated_value += next_token_str
                current_ids.append(next_token_id)
            
            
            # STRING: allow everything except closing quote
            elif param_type == "string":

                valid_ids = []
                for token_str, token_id in vocab.token_to_id.items():
                    if token_str.strip() != '"':
                        valid_ids.append(token_id)

                next_token_id = greedy_next_token(logits_np, valid_ids)
                next_token_str = vocab.id_to_token[next_token_id]

                # closing quote means string is complete
                if next_token_str.strip() == '"':
                    break

                generated_value += next_token_str
                current_ids.append(next_token_id)

            # UNKNOWN TYPE: treat as string to avoid crash
            else:
                print(
                    f"[decoder] WARNING: unknown type '{param_type}' "
                    f"for '{param_name}' — treating as string"
                )

                valid_ids = []
                for token_str, token_id in vocab.token_to_id.items():
                    if token_str.strip() != '"':
                        valid_ids.append(token_id)

                next_token_id = greedy_next_token(logits_np, valid_ids)
                next_token_str = vocab.id_to_token[next_token_id]

                if next_token_str.strip() == '"':
                    break

                generated_value += next_token_str
                current_ids.append(next_token_id)

        # convert generated text to correct Python type
        if param_type == "number":
            try:
                extracted[param_name] = float(generated_value)
            except ValueError:
                extracted[param_name] = 0.0
        else:
            extracted[param_name] = generated_value.strip()

    return extracted