"""Throwaway test script — DO NOT COMMIT."""
import json
from llm_sdk.llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

# --- vocab inspection (you already did this part) ---
vocab_path = model.get_path_to_vocab_file()
print("vocab path:", vocab_path)

# --- Day 4 smoke test: encode -> logits -> numpy -> shape ---
text = "What is the sum of 2 and 3?"

input_ids = model.encode(text)
print("\ninput_ids:", input_ids)
print("length:", len(input_ids))

logits = model.get_logits_from_input_ids(input_ids)
print("\nlogits type:", type(logits))

logits_np = logits.detach().numpy()
print("logits_np type:", type(logits_np))
print("shape:", logits_np.shape)
print("min:", logits_np.min(), "max:", logits_np.max())