import json
from pathlib import Path
from typing import Any

from  src.models import FunctionDefinition, TestPrompt


def load_functions_definition(path: str | Path) -> list[FunctionDefinition]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [FunctionDefinition(**item) for item in data]

    except FileNotFoundError:
        print(f"[io_handler] ERROR: Functions definition file not found: '{path}'")
        return []

    except json.JSONDecodeError as e:
        print(f"[io_handler] ERROR: Invalid JSON in '{path}': {e}")
        return []

    except Exception as e:
        print(f"[io_handler] ERROR: Unexpected error loading functions definition: {e}")
        return []


def load_test_prompts(path: str | Path) -> list[TestPrompt]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [TestPrompt(**item) for item in data]

    except FileNotFoundError:
        print(f"[io_handler] ERROR: Test prompts file not found: '{path}'")
        return []

    except json.JSONDecodeError as e:
        print(f"[io_handler] ERROR: Invalid JSON in '{path}': {e}")
        return []

    except Exception as e:
        print(f"[io_handler] ERROR: Unexpected error loading test prompts: {e}")
        return []


def save_results(results: list[Any], path: str | Path) -> None:
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = [
            r.model_dump() if hasattr(r, "model_dump") else r
            for r in results
        ]

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"[io_handler] Results saved to '{path}' ({len(results)} entries)")

    except OSError as e:
        print(f"[io_handler] ERROR: Could not write to '{path}': {e}")
        raise

    except Exception as e:
        print(f"[io_handler] ERROR: Unexpected error saving results: {e}")
        raise