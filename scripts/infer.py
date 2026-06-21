#!/usr/bin/env python
"""Generate an answer from the base model + the trained LoRA adapter.

    python scripts/infer.py --prompt "What is an emergency fund?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finetune.data import SYSTEM_PROMPT  # noqa: E402


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"])
    model = PeftModel.from_pretrained(base, cfg["output_dir"])
    print(generate(model, tokenizer, args.prompt))


if __name__ == "__main__":
    main()
