import argparse
from src.io_handler import load_functions_definition, load_test_prompts
from src.pipeline import smoke_test

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help = "Path to the functions definition file"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input file"
    )
    parser.add_argument(
        "--output",
        default = "data/output/function_calls.json",
        help = "Path to the output file"
    )
    return parser.parse_args() #returun name space with the arguments as attributes

def main() -> None:
    args = parse_args()

    functions = load_functions_definition(args.functions_definition)
    print(f"\n[main] Loaded {len(functions)} function definitions:")
    for fn in functions:
        print(f"  - {fn.name}: {fn.description}")

    prompts = load_test_prompts(args.input)
    print(f"\n[main] Loaded {len(prompts)} test prompts:")
    for p in prompts:
        print(f"  - {p.prompt}")


if __name__ == "__main__":
    # main()
    smoke_test()