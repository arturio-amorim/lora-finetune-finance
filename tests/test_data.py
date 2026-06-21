"""Tests for dataset loading/formatting (pure — no ML deps needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from finetune.data import SYSTEM_PROMPT, load_examples, render_alpaca, to_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_dataset_loads():
    examples = load_examples(str(ROOT / "data" / "finance_sft.jsonl"))
    assert len(examples) >= 15
    assert all("instruction" in e and "output" in e for e in examples)


def test_to_messages_structure():
    ex = {"instruction": "What is liquidity?", "output": "How fast an asset becomes cash."}
    msgs = to_messages(ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"] == "What is liquidity?"
    assert msgs[2]["content"] == "How fast an asset becomes cash."


def test_to_messages_merges_input():
    ex = {"instruction": "Summarize", "input": "Long text here", "output": "Short."}
    msgs = to_messages(ex)
    assert "Long text here" in msgs[1]["content"]
    assert msgs[1]["content"].startswith("Summarize")


def test_render_alpaca_without_input():
    ex = {"instruction": "Define CAGR", "output": "Average yearly growth."}
    text = render_alpaca(ex)
    assert "### Instruction:" in text and "### Response:" in text
    assert "### Input:" not in text


def test_load_rejects_missing_fields(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"instruction": "no output here"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_examples(str(bad))
