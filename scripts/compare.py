#!/usr/bin/env python
"""Before/after: base model vs the LoRA-finetuned model on the same prompts.

    python scripts/compare.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finetune.data import SYSTEM_PROMPT  # noqa: E402

PROMPTS = [
    "What is an emergency fund and how big should it be?",
    "Explain compound interest simply.",
    "Should I pay off debt or invest first?",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 160) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"])
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(cfg["model"]), cfg["output_dir"]
    )

    for prompt in PROMPTS:
        print("=" * 80)
        print("PROMPT:", prompt)
        print("\n[BASE]\n", generate(base, tokenizer, prompt))
        print("\n[FINE-TUNED]\n", generate(tuned, tokenizer, prompt))
        print()


if __name__ == "__main__":
    main()
