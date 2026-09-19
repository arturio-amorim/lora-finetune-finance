# 🎛️ LoRA Fine-Tuning — Finance Assistant

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-ff6f00)
![HF](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%C2%B7%20TRL-yellow)

Fine-tunes a **small open LLM** (default `Qwen2.5-0.5B-Instruct`) with **LoRA**
into a concise **personal-finance assistant**, then compares the model
**before vs after**. Trains on a **free Colab/Kaggle GPU** — the adapter is only
a few MB.

> Part of a 4-project AI portfolio. This one closes the loop: the result can be
> measured with my [llm-eval-observability](https://github.com/arturio-amorim/llm-eval-observability)
> toolkit instead of eyeballing samples.

## ✨ What it shows

- **Parameter-efficient fine-tuning** (LoRA via  PEFT + TRL `SFTTrainer`).
- **Correct SFT formatting** with the model's own chat template.
- **Before/after comparison** on held-out prompts.
- **Pure, unit-tested data layer** (formatting tested without torch).
- **Push-to-Hub** support for sharing the adapter.

## 🚀 How to run

> Training needs a GPU. The default 0.5B model runs on a **free Colab** GPU.
> The data layer + tests run anywhere with no ML deps.

```bash
pip install -e .
pip install -r requirements.txt

# Train (reads config.yaml)
python scripts/train_lora.py --config config.yaml

# Compare base vs fine-tuned on a few prompts
python scripts/compare.py

# Single prompt against the fine-tuned model
python scripts/infer.py --prompt "Explain compound interest simply."
```

Everything is driven by [`config.yaml`](config.yaml) — model, LoRA rank/alpha,
target modules, learning rate, epochs, and optional Hub push.

## 📊 Before / after

The goal is a model that answers finance questions **concisely and consistently**
(1-3 sentences, no rambling). After fine-tuning, run `compare.py` to see the
shift, and score it with the eval toolkit:

| | Base model | Fine-tuned (LoRA) |
|---|---|---|
| Style | verbose / inconsistent | concise, 1-3 sentences |
| Format adherence | low | high |
| Judge score (correctness) | _run to fill_ | _run to fill_ |

*(Numbers land here after a training run — wired to project #3's LLM-as-judge.)*

## 🗂️ Project structure

```
lora-finetune-finance/
├── config.yaml            # model + LoRA + training hyperparams
├── data/finance_sft.jsonl # instruction dataset (20 examples)
├── src/finetune/data.py   # pure data loading/formatting (tested)
├── scripts/
│   ├── train_lora.py      # LoRA SFT (PEFT + TRL)
│   ├── infer.py           # generate from base + adapter
│   └── compare.py         # before/after
└── tests/test_data.py
```

## ✅ Tests

```bash
pytest -q     # data loading + chat formatting; no GPU/torch needed
```

## 🧭 Roadmap

- [x] LoRA SFT pipeline (config-driven) + before/after compare
- [x] Pure, tested data layer + bundled dataset
- [ ] Run a training pass and publish the adapter to the HF Hub
- [ ] Quantitative before/after via the eval toolkit (project #3)
- [ ] Try QLoRA (4-bit) to fit larger base models

## 📄 License

MIT — see [LICENSE](LICENSE). Base models keep their own licenses.

---

Built by **Arturio Amorim Sobrinho** — AI/LLM Engineer.
[GitHub](https://github.com/arturio-amorim) · [LinkedIn](https://www.linkedin.com/in/arturio-amorim-33b60736/)
