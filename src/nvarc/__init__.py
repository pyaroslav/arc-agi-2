"""Vendored + adapted NVARC (ARC Prize 2025 winner) inference code.

`arc_loader` and `selection` are vendored ~verbatim from NVARC's open-sourced
Kaggle notebook `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue` (open-sourcing
is a competition requirement; we fork it per the ARC Prize 2026 rules). `solver_local`
replaces their unsloth layer with transformers+peft so it runs on a single
RTX 5090 (Blackwell) for local development and reproduction.

Original authors: Ivan Sorokin & Jean-François Puget (NVARC / NVIDIA).
"""
