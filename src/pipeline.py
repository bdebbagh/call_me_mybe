from llm_sdk import Small_LLM_Model  # type: ignore
from src.decoder import VocabularyManager, select_function_name
from src.decoder import extract_arguments
from src.models import FunctionDefinition, TestPrompt, FunctionCall
from src.io_handler import save_results


def run_pipeline(
    prompts: list[TestPrompt],
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: VocabularyManager,
    output_path: str,
) -> None:
    results: list[FunctionCall] = []

    for i, test_prompt in enumerate(prompts):
        print(f"\n[pipeline] Processing prompt {i + 1}/{len(prompts)}: "
              f"{test_prompt.prompt}")

        try:
            function_name = select_function_name(
                test_prompt.prompt, functions, model, vocab
            )
            print(f"[pipeline] Selected function: {function_name}")

            parameters = extract_arguments(
                test_prompt.prompt, function_name, functions, model, vocab
            )
            print(f"[pipeline] Extracted parameters: {parameters}")

            result = FunctionCall(
                prompt=test_prompt.prompt,
                name=function_name,
                parameters=parameters,
            )
            results.append(result)

        except Exception as e:
            print(f"[pipeline] ERROR on prompt '{test_prompt.prompt}': {e}")
            continue

    save_results(results, output_path)
    print(f"\n[pipeline] Done. {len(results)}/{len(prompts)} prompts "
          f"processed.")
