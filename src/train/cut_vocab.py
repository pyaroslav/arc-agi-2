"""Reduce a fresh base model to the 16-token ARC grid vocabulary (NVARC's trick).

A fresh open base (e.g. Qwen3-8B) has a ~150k-token vocab; ARC grids only need 16
tokens. We slice the input embedding and LM head down to 16 rows — initialized from
the base's embeddings of the corresponding source tokens — set vocab_size=16, and
ship our `ArcTokenizer` as the tokenizer. This makes full fine-tuning far cheaper
(tiny softmax/embedding) and matches the released NVARC checkpoint layout so the
existing TTFT->decode->vote pipeline and `ArcTokenizer` work unchanged.

  python -m train.cut_vocab --base Qwen/Qwen3-8B --out models/qwen3_8b_grids16
"""
from __future__ import annotations

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# our 16-token id -> a (string, fallback string) to initialize each row from the base
# ids match src/nvarc/solver_local.py ArcTokenizer: 0-9 digits, 10 \n, 11 user,
# 12 assistant, 13 pad, 14 im_start, 15 im_end.
SOURCES = {
    **{i: (str(i), str(i)) for i in range(10)},
    10: ("\n", "Ċ"),
    11: ("user", "Ġuser"),
    12: ("assistant", "Ġassistant"),
    13: ("<|endoftext|>", "<|im_end|>"),
    14: ("<|im_start|>", "<|im_start|>"),
    15: ("<|im_end|>", "<|endoftext|>"),
}


def _src_id(tok, primary, fallback):
    for s in (primary, fallback):
        try:
            ids = tok.encode(s, add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) == 1:
            return ids[0]
        cid = tok.convert_tokens_to_ids(s)
        if isinstance(cid, int) and cid >= 0 and cid != getattr(tok, "unk_token_id", -1):
            return cid
    return tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    emb = model.get_input_embeddings().weight.data            # [V, H]
    H = emb.shape[1]
    tie = bool(getattr(model.config, "tie_word_embeddings", False))
    head = None if tie else model.get_output_embeddings().weight.data

    new_emb = torch.zeros(16, H, dtype=emb.dtype)
    new_head = None if tie else torch.zeros(16, head.shape[1], dtype=head.dtype)
    for i in range(16):
        sid = _src_id(tok, *SOURCES[i])
        new_emb[i] = emb[sid]
        if new_head is not None:
            new_head[i] = head[sid]
        print(f"  id {i:2d} <- base token id {sid}", flush=True)

    model.resize_token_embeddings(16)
    model.get_input_embeddings().weight.data.copy_(new_emb)
    if not tie:
        model.get_output_embeddings().weight.data.copy_(new_head)
    model.config.vocab_size = 16
    model.config.bos_token_id, model.config.eos_token_id = 13, 15

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    # ship a minimal tokenizer marker; inference uses src.nvarc.solver_local.ArcTokenizer
    with open(os.path.join(args.out, "ARC_TOKENIZER.txt"), "w") as f:
        f.write("Use nvarc.solver_local.ArcTokenizer (16-token grid vocab). vocab_size=16.\n")
    print(f"saved 16-token base to {args.out} (vocab 16, hidden {H}, tied={tie})", flush=True)


if __name__ == "__main__":
    main()
