"""Merge a trained LoRA adapter into its base -> a standalone 16-token checkpoint
(copies tokenizer/config so repro_nvarc.py / kaggle_submit.py can load it)."""
import argparse
import glob
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from transformers import AutoModelForCausalLM
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/nvarc/qwen3_4b_sft")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out)
    # copy tokenizer/config sidecars from base
    for f in glob.glob(os.path.join(args.base, "*.json")) + glob.glob(os.path.join(args.base, "*.txt")):
        if "model.safetensors.index" in f or os.path.basename(f) == "config.json":
            continue
        shutil.copy(f, args.out)
    print(f"merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
