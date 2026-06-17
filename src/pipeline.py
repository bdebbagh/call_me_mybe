import numpy as np
from llm_sdk import Small_LLM_Model

from src.decoder import VocabularyManager, greedy_next_token


def smoke_test() -> None:
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabularyManager(path=vocab_path)
    print(f"[pipeline] Loaded vocabulary with {len(vocab.token_to_id)} tokens")

    text = "What is the sum of 2 and 3?"
    input_ids = model.encode(text)
    print(input_ids)
    input_ids = input_ids[0].tolist()   # tensor([[...]]) → [3838, 374, 279, ...]
    print(f"[pipeline] Encoded '{text}' -> {input_ids}")

    logits = model.get_logits_from_input_ids(input_ids)
    logits_np = np.array(logits)
    print(f"[pipeline] Logits shape: {logits_np.shape}")
    print(f"[pipeline] Logits min/max: {logits_np.min():.3f} / {logits_np.max():.3f}")

    # Dummy constrained-decoding test with a made-up constraint
    valid_ids = vocab.get_tokens_starting_with("2") + vocab.get_tokens_starting_with("3")
    if valid_ids:
        chosen_id = greedy_next_token(logits_np, valid_ids)
        chosen_str = vocab.id_to_token.get(chosen_id, "<unknown>")
        print(f"[pipeline] greedy_next_token chose id={chosen_id} -> '{chosen_str}'")
    else:
        print("[pipeline] No valid_ids found for test prefixes '2'/'3'")