*This project has been created as part of the 42 curriculum by bdebbagh.*

# call me maybe — Introduction to Function Calling in LLMs

## Description

This project implements a **function calling system** using a small Large Language Model (Qwen3-0.6B). Given a natural language prompt like `"What is the sum of 2 and 3?"`, the system identifies the correct function to call and extracts the arguments from the prompt, outputting a structured JSON result:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The key challenge is reliability. Small language models fail to produce valid JSON ~70% of the time when prompted directly. This project solves that using **constrained decoding** — a technique that controls the model's output token by token to guarantee 100% valid, schema-compliant JSON every time.

---

## Instructions

### Requirements

- Python 3.10 or later
- `uv` package manager

### Installation

```bash
git clone <your_repo_url>
cd call_me_mybe
uv sync
```

### Running

```bash
# with default paths
uv run python -m src

# with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other commands

```bash
make install      # install dependencies
make run          # run the project
make lint         # run flake8 and mypy
make clean        # remove cache files
```

---

## Algorithm Explanation

The system uses **constrained decoding** in two passes for each prompt:

### Pass 1 — Function Selection (`select_function_name`)

1. Build a prompt that lists all available functions and ends with `{"name": "`
2. Tokenize the prompt and feed it to the model
3. At each generation step:
   - Get 150,000 logit scores from the model
   - Check if what has been generated so far is already a complete function name → return it
   - Find all tokens where `(generated_so_far + token)` is still the beginning of a valid function name (prefix matching)
   - Set all other tokens to `-infinity`
   - Pick the token with the highest remaining score
4. Repeat until a complete valid function name is produced

### Pass 2 — Argument Extraction (`extract_arguments`)

For each parameter of the chosen function:

1. Build a prompt ending just before the parameter value
2. Run the generation loop:
   - **Number parameters**: only allow tokens containing digits, `.`, `-`, `e`, `E`. Stop when the model naturally wants to write a non-digit character
   - **String parameters**: allow all tokens freely. Stop when the model produces a closing quote `"`
3. Convert the generated text to the correct Python type (`float` for numbers, `str` for strings)

---

## Design Decisions

**Why constrained decoding instead of prompting?**
Prompting a small model to produce JSON is unreliable (~30% success rate). Constrained decoding guarantees valid output by controlling every token.

**Why two separate passes?**
Separating function selection from argument extraction makes each pass simpler and more reliable. The model focuses on one task at a time.

**Why check the model's natural top choice before constraining for numbers?**
When the model naturally wants to write a non-digit (like `,`), it signals the number is complete. Using this as an early exit prevents forcing extra digits onto a finished number.

**Why allow the model to pick freely for strings?**
For strings, the model knows best when to stop — it will naturally produce a closing quote `"` when the value is complete. Constraining string tokens would prevent the model from stopping.

**Why pydantic for data models?**
Automatic validation catches missing fields and wrong types immediately at the point of creation, not silently later in the pipeline.

---

## Performance Analysis

Results on the provided test set (11 prompts):

| Prompt | Function Selected | Parameters Correct |
|--------|------------------|-------------------|
| What is the sum of 2 and 3? | fn_add_numbers  | a=2.0, b=3.0  |
| What is the sum of 265 and 345? | fn_add_numbers  | a=265.0, b=345.0  |
| Greet shrek | fn_greet  | name=shrek  |
| Greet john | fn_greet  | name=john  |
| Reverse the string 'hello' | fn_reverse_string  | s=hello |
| Reverse the string 'world' | fn_reverse_string | s=world  |
| What is the square root of 16? | fn_get_square_root  | a=16.0  |
| Calculate the square root of 144 | fn_get_square_root  | a=144.0  |
| Replace all numbers... | fn_substitute_string_with_regex  | all params  |
| Replace all vowels... | fn_substitute_string_with_regex  | all params  |
| Substitute the word 'cat'... | fn_substitute_string_with_regex  | all params  |

**Accuracy: 11/11 (100%)**
**JSON validity: 100%** — every output is parseable

**Speed**: running on CPU, approximately 3-5 minutes for all 11 prompts depending on hardware.

---

## Challenges Faced

**1. String loop running forever**
The string generation loop blocked the closing quote `"` from valid tokens, so the model could never pick it to stop. Fixed by removing the constraint on strings entirely and letting the model pick freely.

**2. Tokens containing spaces (Ġ character)**
The tokenizer encodes spaces as `Ġ` inside token strings. When concatenating tokens, these appeared as `Ġ` characters in the output. Fixed by replacing `Ġ` with a real space after generation.

**3. Number loop never stopping**
Without a stopping condition, the number loop ran until max_steps because digit tokens are always available. Fixed by checking the model's unconstrained top choice first — if it wants a non-digit, the number is complete.

**4. Torch GPU library missing on CPU machine**
The default torch installation includes GPU libraries that were missing on the development machine. Fixed by installing the CPU-only version of torch via the PyTorch CPU index.

---

## Testing Strategy

1. **Smoke test** (`pipeline.smoke_test`): verifies the model loads, encodes text, produces logits, and basic constrained token selection works.

2. **Function selection test**: ran `select_function_name` on all 11 prompts and verified each returned the correct function name.

3. **Argument extraction test**: ran `extract_arguments` on each prompt after confirming function selection, verified values match expected output.

4. **Full pipeline test**: ran `uv run python -m src` on all 11 prompts and checked the output JSON file.

5. **Edge cases checked**: numbers with multiple digits (265, 345, 144), strings with spaces, regex patterns with special characters.

---

## Example Usage

```bash
# default run
uv run python -m src

# expected output in terminal
[main] Loaded 5 function definitions
[main] Loaded 11 test prompts
[main] Loading model...
[pipeline] Processing prompt 1/11: What is the sum of 2 and 3?
[pipeline] Selected function: fn_add_numbers
[pipeline] Extracted parameters: {'a': 2.0, 'b': 3.0}
...
[pipeline] Done. 11/11 prompts processed.

# check output file
cat data/output/function_calls.json
```

Example output file:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

---

## Resources

### Documentation
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [PyTorch Documentation](https://pytorch.org/docs)

### Articles
- [Constrained Decoding for LLMs](https://huggingface.co/blog/constrained-beam-search)
- [Byte Pair Encoding Tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Function Calling in LLMs](https://platform.openai.com/docs/guides/function-calling)

### AI Usage
AI  was used during this project for:
- **Theory understanding**: explaining how tokenization, logits, and autoregressive generation work from scratch
- **Debugging**: identifying the string loop infinite bug and the Ġ character issue
- **Documentation**: helping structure this README

All code was understood, reviewed, and is fully explainable by the author.