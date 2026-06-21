"""Dataset loading + formatting (pure Python, no ML deps — so it's testable)."""
from __future__ import annotations

import json
from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a concise personal-finance assistant. Answer clearly and correctly "
    "in 1-3 sentences. Do not give individualized investment advice; explain "
    "concepts and trade-offs."
)


def load_examples(path: str) -> List[dict]:
    """Load an instruction dataset in JSONL ({instruction, input?, output})."""
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    for i, ex in enumerate(examples):
        if "instruction" not in ex or "output" not in ex:
            raise ValueError(f"example {i} must have 'instruction' and 'output'")
    return examples


def _user_text(example: dict) -> str:
    text = example["instruction"].strip()
    extra = (example.get("input") or "").strip()
    return f"{text}\n\n{extra}" if extra else text


def to_messages(example: dict, system: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Chat-format an example (system / user / assistant)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_text(example)},
        {"role": "assistant", "content": example["output"].strip()},
    ]


def render_alpaca(example: dict) -> str:
    """Plain Alpaca-style rendering (fallback when no chat template available)."""
    user = _user_text(example)
    return f"### Instruction:\n{user}\n\n### Response:\n{example['output'].strip()}"
