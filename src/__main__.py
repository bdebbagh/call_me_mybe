import argparse
from src.io_handler import load_functions_definition, load_test_prompts
from src.pipeline import run_pipeline
from src.decoder import VocabularyManager
from llm_sdk import Small_LLM_Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition file"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input file"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to the output file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # load input files
    functions = load_functions_definition(args.functions_definition)
    print(f"\n[main] Loaded {len(functions)} function definitions:")
    for fn in functions:
        print(f"  - {fn.name}: {fn.description}")

    prompts = load_test_prompts(args.input)
    print(f"\n[main] Loaded {len(prompts)} test prompts:")
    for p in prompts:
        print(f"  - {p.prompt}")

    # load model and vocabulary
    print("\n[main] Loading model...")
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabularyManager(path=vocab_path)
    print(f"[main] Loaded vocabulary: {len(vocab.token_to_id)} tokens")

    # run the full pipeline
    run_pipeline(prompts, functions, model, vocab, args.output)


if __name__ == "__main__":
    main()