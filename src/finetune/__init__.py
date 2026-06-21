"""finetune — LoRA SFT of a small open LLM into a finance assistant.

The data layer (:mod:`finetune.data`) is pure-Python and unit-tested; the
training/inference scripts pull in the heavy ML stack only when run.
"""
from .data import load_examples, render_alpaca, to_messages

__version__ = "0.1.0"
__all__ = ["load_examples", "to_messages", "render_alpaca", "__version__"]
