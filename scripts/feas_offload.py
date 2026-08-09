"""Feasibility test: FULL fine-tuning with the optimizer state offloaded to CPU RAM
(pure PyTorch, no DeepSpeed/bnb build). Measures VRAM, RAM, and seconds/step so we
can project days/epoch for a real run, and whether 8B fits this box.

  PYTHONPATH=src python scripts/feas_offload.py --base data/nvarc/qwen3_4b_sft --steps 6
"""
import argparse
import os
import sys
import time

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from transformers import AutoModelForCausalLM
from train.sft_data import iter_examples, pack
from train.sft_train import collate, PAD_ID


class CPUOffloadAdamW:
    """fp32 master + Adam moments live in CPU RAM; GPU only holds bf16 weights+grads."""
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, wd=0.0,
                 state_dtype=torch.float32):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.params = [p for p in params if p.requires_grad]
        self.master = [p.detach().float().cpu() for p in self.params]
        self.m = [torch.zeros(p.shape, dtype=state_dtype) for p in self.params]
        self.v = [torch.zeros(p.shape, dtype=state_dtype) for p in self.params]
        self.t = 0
        gb = sum(x.numel() for x in self.master) * (4 + 2 * (2 if state_dtype == torch.bfloat16 else 4)) / 1e9
        print(f"  optimizer state in RAM: ~{gb:.1f} GB ({len(self.params)} tensors, "
              f"moments={state_dtype})", flush=True)

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach().float().to("cpu", non_blocking=True)
            m, v, mp = self.m[i], self.v[i], self.master[i]
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (v.float().sqrt() / (bc2 ** 0.5)).add_(self.eps)
            if self.wd:
                mp.mul_(1 - self.lr * self.wd)
            mp.addcdiv_(m.float(), denom, value=-self.lr / bc1)
            p.data.copy_(mp.to(p.dtype).to(p.device, non_blocking=True))

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/nvarc/qwen3_4b_sft")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--bf16-moments", action="store_true", help="store Adam m,v in bf16 (halves RAM)")
    args = ap.parse_args()

    proc = psutil.Process()
    print(f"loading {args.base} (bf16, full-FT) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e9:.2f}B", flush=True)

    data = pack([e for e in iter_examples(["data/arc-agi-2/training"], n_aug=1,
                                          pairs_per_example=4, max_seq_len=args.seq, seed=1)][:40],
                max_seq_len=args.seq, pad_id=PAD_ID)
    opt = CPUOffloadAdamW(model.parameters(), lr=1e-5,
                          state_dtype=torch.bfloat16 if args.bf16_moments else torch.float32)
    torch.cuda.reset_peak_memory_stats()

    fwd_t, opt_t = [], []
    for s in range(args.steps):
        batch = [data[s % len(data)]]
        ii, att, lab = collate(batch, "cuda")
        t0 = time.time()
        loss = model(input_ids=ii, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.cuda.synchronize(); t1 = time.time()
        opt.step(); opt.zero_grad()
        t2 = time.time()
        fwd_t.append(t1 - t0); opt_t.append(t2 - t1)
        print(f"  step {s}: loss={loss.item():.3f} fwd+bwd={t1-t0:.1f}s opt={t2-t1:.1f}s "
              f"VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB RAM={proc.memory_info().rss/1e9:.0f}GB",
              flush=True)

    fb = np.median(fwd_t[1:] or fwd_t); op = np.median(opt_t[1:] or opt_t)
    step_s = fb + op
    print(f"\n==== FEASIBILITY ({n_params/1e9:.1f}B full-FT, CPU-offload optimizer) ====")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB (of ~30 usable)")
    print(f"  RAM (this proc): {proc.memory_info().rss/1e9:.0f} GB | system avail: {psutil.virtual_memory().available/1e9:.0f} GB")
    print(f"  median fwd+bwd: {fb:.1f}s | optimizer(CPU): {op:.1f}s | per-step: {step_s:.1f}s")
    # project: ~50k packed sequences/epoch for the synthetic corpus
    for nseq, lbl in [(50000, "synthetic corpus ~1 epoch")]:
        print(f"  projected {lbl} ({nseq} seq, bs1): {nseq*step_s/3600:.0f}h = {nseq*step_s/86400:.1f} days")


if __name__ == "__main__":
    main()
