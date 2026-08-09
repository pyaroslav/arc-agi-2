"""Test-time training (TTFT): fine-tune a fresh per-task LoRA on the task's own
demonstration pairs, using a leave-one-out objective under random augmentations.

For each step we sample an augmentation, transform the task, hold out one demo
pair, and train the model to produce that pair's output given the others as
context (mirroring the test-time structure). This is the per-task adaptation that
drives ARC-AGI-2 scores (Akyürek TTT / NVARC TTFT).
"""
from __future__ import annotations

from typing import List
import numpy as np
import torch

from .model import ModelBundle, add_lora, build_supervised_example
from .data import Task, sample_augmentations


def train_task_lora(bundle: ModelBundle, task: Task, *, steps: int, lr: float,
                    lora_r: int, lora_alpha: int, lora_dropout: float,
                    target_modules: List[str], grad_checkpointing: bool = False,
                    seed: int = 0, log_every: int = 0) -> ModelBundle:
    """Attach + train a per-task LoRA in place. Returns the bundle (model wrapped)."""
    bundle = add_lora(bundle, lora_r, lora_alpha, lora_dropout, target_modules)
    model = bundle.model
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    demos = task["train"]
    n = len(demos)
    rng = np.random.default_rng(seed)
    # Pre-sample one augmentation per step (distinct random aug each step, NVARC-style).
    augs = sample_augmentations(steps, rng, n_demos=n, include_identity=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for step in range(steps):
        aug = augs[step % len(augs)]
        atask = aug.apply_task({"train": demos, "test": [{"input": demos[0]["input"]}]})
        apairs = atask["train"]
        held = int(rng.integers(0, n)) if n > 1 else 0
        context = [apairs[j] for j in range(n) if j != held] if n > 1 else apairs
        q = apairs[held]
        input_ids, labels = build_supervised_example(bundle, context, q["input"], q["output"])
        out = model(input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  ttft step {step:4d}/{steps}  loss={loss.item():.4f}")

    model.eval()
    return bundle


def remove_lora(bundle: ModelBundle) -> ModelBundle:
    """Unwrap the PEFT model back to the base (so the next task starts clean)."""
    if hasattr(bundle.model, "unload"):
        bundle.model = bundle.model.unload()  # merge-free: returns base model
    elif hasattr(bundle.model, "get_base_model"):
        bundle.model = bundle.model.get_base_model()
    return bundle
