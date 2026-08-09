"""CPU-offloaded AdamW: fp32 master + Adam moments live in CPU RAM, the GPU only
holds bf16 weights + grads. Enables FULL fine-tuning of 7-8B on a 32GB GPU (the
optimizer state, not VRAM, is the constraint -> offloaded to RAM). No DeepSpeed/bnb
build needed (pure PyTorch), so it works on Blackwell.

Per-tensor in-place update (small temporaries) — measured faster than `_foreach`,
which allocates ~2x the optimizer state in transient buffers per step. The step is
memory-bandwidth-bound (~7.5s for 4B / ~15s for 8B); use gradient accumulation to
amortize it (at grad_accum=16 a 4B epoch is ~15h).

RAM ~= 12 bytes/param (fp32 master+m+v): 4B->~44GB, 8B->~96GB. lr is a settable attr
so the training loop can drive a manual LR schedule.
"""
from __future__ import annotations
import torch


class CPUOffloadAdamW:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 moment_dtype=torch.float32):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.master = [p.detach().float().cpu() for p in self.params]            # fp32
        self.m = [torch.zeros(p.shape, dtype=moment_dtype) for p in self.params]  # bf16 halves RAM
        self.v = [torch.zeros(p.shape, dtype=moment_dtype) for p in self.params]
        self.t = 0
        bpp = 4 + 2 * (2 if moment_dtype == torch.bfloat16 else 4)               # master + m + v bytes/param
        gb = sum(x.numel() for x in self.master) * bpp / 1e9
        print(f"  CPUOffloadAdamW: {len(self.params)} tensors, ~{gb:.0f}GB optimizer state in RAM "
              f"(moments={moment_dtype})", flush=True)

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        step_size = self.lr / bc1
        bf16 = self.m[0].dtype == torch.bfloat16
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach().to("cpu", torch.float32)
            m, v, mp = self.m[i], self.v[i], self.master[i]
            mf = m.float() if bf16 else m       # compute in fp32; tiny per-tensor temp
            vf = v.float() if bf16 else v
            mf.mul_(self.b1).add_(g, alpha=1 - self.b1)
            vf.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = vf.sqrt().div_(bc2 ** 0.5).add_(self.eps)
            if self.wd:
                mp.mul_(1 - self.lr * self.wd)
            mp.addcdiv_(mf, denom, value=-step_size)
            if bf16:
                m.copy_(mf); v.copy_(vf)        # store back to bf16
            p.data.copy_(mp.to(p.dtype).to(p.device))

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None
