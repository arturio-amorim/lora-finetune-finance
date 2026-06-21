#!/usr/bin/env python
"""LoRA SFT of a small open LLM on the finance instruction dataset.

Best on a GPU — the free Colab/Kaggle tiers work for the default 0.5B model.

    python scripts/train_lora.py --config config.yaml

Tested with: transformers>=4.44, peft>=0.12, trl>=0.11, datasets>=2.20.
(If your TRL is older, pass `tokenizer=tok` to SFTTrainer instead of
`processing_class`.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finetune.data import load_examples, to_messages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT trainer.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # heavy imports happen only when actually training
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_name = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # format every example with the model's own chat template
    examples = load_examples(cfg["data_path"])
    texts = [tokenizer.apply_chat_template(to_messages(ex), tokenize=False) for ex in examples]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Loaded {len(dataset)} training examples")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    tr = cfg["training"]
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=tr["epochs"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["grad_accum"],
        learning_rate=tr["lr"],
        max_seq_length=tr["max_seq_length"],
        logging_steps=tr["logging_steps"],
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved LoRA adapter -> {cfg['output_dir']}")

    hub = cfg.get("hub", {})
    if hub.get("push") and hub.get("repo_id"):
        trainer.push_to_hub(hub["repo_id"])
        print(f"Pushed adapter -> https://huggingface.co/{hub['repo_id']}")


if __name__ == "__main__":
    main()
