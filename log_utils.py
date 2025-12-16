import json
from typing import Any


def _stringify(content: Any) -> str:
    if content is None:
        return "(empty)"
    if isinstance(content, str):
        stripped = content.strip()
        return stripped if stripped else "(empty)"
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return str(content)


def print_step(title: str, content: Any) -> None:
    """
    Prints a formatted block for a pipeline step so CLI logs stay consistent.
    """
    header = f"{'=' * 20} {title} {'=' * 20}"
    footer = "=" * len(header)
    body = _stringify(content)
    print(f"\n{header}\n{body}\n{footer}\n")
