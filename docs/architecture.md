# Architecture

```mermaid
flowchart TD
    D["finance_sft.jsonl<br/>(instruction, output)"] --> F["format with chat template<br/>finetune.data.to_messages"]
    F --> T["SFT training (TRL)"]
    B["Base model<br/>Qwen2.5-0.5B-Instruct"] --> T
    L["LoRA adapters<br/>(PEFT: q/k/v/o_proj)"] --> T
    T --> A["Trained adapter<br/>outputs/lora-finance"]
    A --> C["compare.py<br/>base vs fine-tuned"]
    A --> H["(optional) push to HF Hub"]
    A -.->|evaluate with| E["llm-eval-observability<br/>(project #3)"]
```

## Why LoRA

Full fine-tuning updates every weight — expensive and memory-hungry. **LoRA**
freezes the base model and trains tiny low-rank adapter matrices on a few
attention projections (`q/k/v/o_proj`). You get most of the benefit while
training <1% of the parameters, so it fits on a free Colab GPU and the
resulting adapter is only a few MB.

## Design choices

- **Small base model by default** (`Qwen2.5-0.5B-Instruct`) so the project is
  reproducible on free hardware; swap the `model` field for a larger one.
- **Chat-template formatting** uses the model's own template, so the fine-tune
  matches how the model is actually prompted at inference.
- **Pure, tested data layer** — formatting is unit-tested without torch.
- **Evaluation closes the loop** — measure the before/after model with
  [llm-eval-observability](https://github.com/arturio-amorim/llm-eval-observability)
  instead of eyeballing a couple of samples.
