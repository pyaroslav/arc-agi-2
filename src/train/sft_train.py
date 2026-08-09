"""SFT / continued-FT of an ARC base on synthetic data (Lever C).

Loads a base (default: NVARC's 16-token Qwen3-4B, for continued-FT), attaches a
LoRA (fits a single 32GB GPU; use --full-ft on cloud), trains with the NVARC recipe
(bf16, grad-checkpointing, cosine LR, sequence-packed completion-only loss), and
saves the adapter/model. The result drops into the existing TTFT->decode->vote
pipeline as a stronger base -> more generation coverage.

  python -m train.sft_train --base data/nvarc/qwen3_4b_sft \
      --sources data/nvarc/synthetic data/arc-agi-2/training \
      --max-examples 4000 --epochs 1 --out outputs/sft_4b_lora
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.sft_data import iter_examples, pack
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

PAD_ID = 13


def build(sources, max_examples, max_seq_len, n_aug, pairs_per_example, seed):
    gen = iter_examples(sources, n_aug=n_aug, pairs_per_example=pairs_per_example,
                        max_seq_len=max_seq_len, seed=seed)
    exs = []
    for ex in gen:
        exs.append(ex)
        if max_examples and len(exs) >= max_examples:
            break
    random.Random(seed).shuffle(exs)
    return pack(exs, max_seq_len=max_seq_len, pad_id=PAD_ID)


def atomic_save(model, out, step):
    """Checkpoint to <out> via tmp-dir + rename: a kill mid-write can never
    corrupt the last good checkpoint. trainer_state.json records the step
    so --resume can continue."""
    tmp, old = out + ".tmp", out + ".old"
    for d in (tmp, old):
        if os.path.exists(d):
            shutil.rmtree(d)
    model.save_pretrained(tmp)
    with open(os.path.join(tmp, "trainer_state.json"), "w") as f:
        json.dump({"step": step}, f)
    if os.path.exists(out):
        os.rename(out, old)
    os.rename(tmp, out)
    if os.path.exists(old):
        shutil.rmtree(old)


def collate(batch, device):
    m = max(len(b["input_ids"]) for b in batch)
    ii, lab, att = [], [], []
    for b in batch:
        p = m - len(b["input_ids"])
        ii.append(b["input_ids"] + [PAD_ID] * p)
        lab.append(b["labels"] + [-100] * p)
        att.append([1] * len(b["input_ids"]) + [0] * p)
    return (torch.tensor(ii, device=device), torch.tensor(att, device=device),
            torch.tensor(lab, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/nvarc/qwen3_4b_sft")
    ap.add_argument("--sources", nargs="+", default=["data/arc-agi-2/training"])
    ap.add_argument("--out", default="outputs/sft_lora")
    ap.add_argument("--max-examples", type=int, default=4000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--n-aug", type=int, default=4)
    ap.add_argument("--pairs-per-example", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--full-ft", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--offload", action="store_true", help="offload optimizer state to CPU RAM (full-FT of 7-8B on a 32GB GPU)")
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (0=only at end); enables spot-instance resume")
    ap.add_argument("--resume", action="store_true", help="continue LoRA training from <out>'s last checkpoint (weights + step; optimizer moments restart). No-op if <out> has no trainer_state.json.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(f"building SFT data from {args.sources} (<= {args.max_examples} examples)...", flush=True)
    data = build(args.sources, args.max_examples, args.max_seq_len, args.n_aug,
                 args.pairs_per_example, args.seed)
    print(f"packed {len(data)} sequences | avg len {int(np.mean([len(d['input_ids']) for d in data]))}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    resume_step = 0
    if args.full_ft:
        model.enable_input_require_grads()
        params = model.parameters()
        print("FULL fine-tune", flush=True)
    else:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
                         use_rslora=True, task_type="CAUSAL_LM")
        state_f = os.path.join(args.out, "trainer_state.json")
        if args.resume and os.path.exists(state_f):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.out, is_trainable=True)
            resume_step = json.load(open(state_f))["step"]
            print(f"resumed adapter from {args.out} @ step {resume_step}", flush=True)
        else:
            model = get_peft_model(model, cfg)
        model.enable_input_require_grads()
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"LoRA r={args.lora_r}; trainable params {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)

    model.train()
    steps_per_epoch = max(1, len(data) // (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    if args.offload:
        from train.offload_opt import CPUOffloadAdamW
        import math
        opt = CPUOffloadAdamW(params, lr=args.lr, weight_decay=0.0,
                              moment_dtype=torch.bfloat16)   # ~64GB for 8B (fits RAM w/o freeing VMs)
        warmup = max(1, int(0.03 * total_steps))
        def lr_at(s):
            if s < warmup:
                return args.lr * s / warmup
            prog = (s - warmup) / max(1, total_steps - warmup)
            return 0.5 * args.lr * (1 + math.cos(math.pi * prog))
        sched = None
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
        for _ in range(resume_step):
            sched.step()
    print(f"optimizer steps: {total_steps}", flush=True)

    t0 = time.time(); step = resume_step; micro = 0; running = 0.0
    order = list(range(len(data)))
    for epoch in range(int(np.ceil(args.epochs))):
        random.Random(args.seed + epoch).shuffle(order)
        done_in_epoch = step - epoch * steps_per_epoch   # >0 only when resuming
        if done_in_epoch >= steps_per_epoch:
            continue
        start_bi = max(0, done_in_epoch) * args.batch_size * args.grad_accum
        for bi in range(start_bi, len(order), args.batch_size):
            batch = [data[order[j]] for j in range(bi, min(bi + args.batch_size, len(order)))]
            ii, att, lab = collate(batch, "cuda")
            loss = model(input_ids=ii, attention_mask=att, labels=lab).loss / args.grad_accum
            loss.backward(); running += loss.item() * args.grad_accum; micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if sched is None:
                    opt.lr = lr_at(step)
                opt.step()
                if sched is not None:
                    sched.step()
                opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    cur_lr = sched.get_last_lr()[0] if sched is not None else opt.lr
                    print(f"  step {step}/{total_steps} loss={running/args.grad_accum/10:.4f} "
                          f"lr={cur_lr:.2e} ({(time.time()-t0)/60:.1f}m)", flush=True)
                    running = 0.0
                if args.save_every and step % args.save_every == 0:
                    atomic_save(model, args.out, step)
                    print(f"  checkpoint @ step {step} -> {args.out}", flush=True)
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    atomic_save(model, args.out, step)
    print(f"saved to {args.out} ({(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
