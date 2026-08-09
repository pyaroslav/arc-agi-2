"""Lever C: supervised fine-tuning (SFT) of a stronger base on synthetic ARC data,
to raise the generation *coverage* that the inference-time levers cannot. Reuses the
16-token grid tokenizer + NVARC format from src/nvarc."""
