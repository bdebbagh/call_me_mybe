import sys
sys.path.insert(0, '.')

from llm_sdk import Small_LLM_Model
from src.decoder import VocabularyManager, select_function_name
from src.io_handler import load_functions_definition

# 1. load functions
functions = load_functions_definition("data/input/functions_definition.json")
print(f"Loaded {len(functions)} functions:")
for fn in functions:
    print(f"  - {fn.name}")

# 2. load model
print("\nLoading model...")
model = Small_LLM_Model()

# 3. load vocabulary
vocab_path = model.get_path_to_vocab_file()
vocab = VocabularyManager(path=vocab_path)
print(f"Loaded vocabulary: {len(vocab.token_to_id)} tokens")

# 4. test 3 prompts
test_prompts = [
    "What is the sum of 2 and 3?",
    "Greet shrek",
    "Reverse the string hello",
]

print("\n--- RESULTS ---")
for prompt in test_prompts:
    result = select_function_name(prompt, functions, model, vocab)
    print(f"prompt : {prompt}")
    print(f"result : {result}")
    print()









