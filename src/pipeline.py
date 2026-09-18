import numpy as np
from llm_sdk import Small_LLM_Model

from src.decoder import VocabularyManager, greedy_next_token, select_function_name, extract_arguments
from src.models import FunctionDefinition, TestPrompt, FunctionCall
from src.io_handler import save_results


def smoke_test() -> None:
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabularyManager(path=vocab_path)
    print(f"[pipeline] Loaded vocabulary with {len(vocab.token_to_id)} tokens")

    text = "What is the sum of 2 and 3?"
    input_ids = model.encode(text)
    print(input_ids)
    input_ids = input_ids[0].tolist()
    print(f"[pipeline] Encoded '{text}' -> {input_ids}")

    logits = model.get_logits_from_input_ids(input_ids)
    logits_np = np.array(logits)
    print(f"[pipeline] Logits shape: {logits_np.shape}")
    print(f"[pipeline] Logits min/max: {logits_np.min():.3f} / {logits_np.max():.3f}")

    valid_ids = vocab.get_tokens_starting_with("2") + vocab.get_tokens_starting_with("3")
    if valid_ids:
        chosen_id = greedy_next_token(logits_np, valid_ids)
        chosen_str = vocab.id_to_token.get(chosen_id, "<unknown>")
        print(f"[pipeline] greedy_next_token chose id={chosen_id} -> '{chosen_str}'")
    else:
        print("[pipeline] No valid_ids found for test prefixes '2'/'3'")


def run_pipeline(
    prompts: list[TestPrompt],
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: VocabularyManager,
    output_path: str,
) -> None:
    """Process all prompts and save results to output file."""

    results: list[FunctionCall] = []

    for i, test_prompt in enumerate(prompts):
        print(f"\n[pipeline] Processing prompt {i + 1}/{len(prompts)}: {test_prompt.prompt}")

        try:
            # step 1 — choose which function to call
            function_name = select_function_name(
                test_prompt.prompt, functions, model, vocab
            )
            print(f"[pipeline] Selected function: {function_name}")

            # step 2 — extract the argument values
            parameters = extract_arguments(
                test_prompt.prompt, function_name, functions, model, vocab
            )
            print(f"[pipeline] Extracted parameters: {parameters}")

            # step 3 — build the result object
            result = FunctionCall(
                prompt=test_prompt.prompt,
                name=function_name,
                parameters=parameters,
            )
            results.append(result)

        except Exception as e:
            print(f"[pipeline] ERROR on prompt '{test_prompt.prompt}': {e}")
            continue

    # step 4 — save all results
    save_results(results, output_path)
    print(f"\n[pipeline] Done. {len(results)}/{len(prompts)} prompts processed.")

