"""Adapter: sorokin's arrow corpus (puzzle_name, messages[]) -> packed SFT data.

Renders each row's chat turns in the exact QwenFormatter template and labels
completion-only (assistant spans). Streams shards; caps rows via max_rows.
"""
from __future__ import annotations

import glob
import random

import numpy as np
import pyarrow as pa

from nvarc.solver_local import ArcTokenizer, build_completion_labels
from train.sft_data import pack

PAD_ID = 13


def render(messages) -> str:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append("<|im_start|>user\n" + m["content"] + "<|im_end|>")
        else:
            out.append("<|im_start|>assistant\n" + m["content"] + "<|im_end|>")
    return "".join(out)


def build_arrow(sources, max_rows, max_seq_len, seed=0):
    tok = ArcTokenizer()
    files = []
    for s in sources:
        files += sorted(glob.glob(s.rstrip("/") + "/data-*.arrow"))
    rng = random.Random(seed)
    rng.shuffle(files)
    exs, n_seen = [], 0
    for f in files:
        table = pa.ipc.open_stream(open(f, "rb")).read_all()
        idx = list(range(table.num_rows))
        rng.shuffle(idx)
        rows = table.to_pylist()
        for i in idx:
            if max_rows and n_seen >= max_rows:
                break
            n_seen += 1
            try:
                ids = tok.encode(render(rows[i]["messages"]))
            except Exception:
                continue
            if len(ids) > max_seq_len:
                continue
            labels = build_completion_labels(ids)
            if all(l == -100 for l in labels):
                continue
            exs.append((ids, labels))
        del table, rows
        if max_rows and n_seen >= max_rows:
            break
    rng.shuffle(exs)
    print(f"arrow corpus: {n_seen} rows seen, {len(exs)} usable examples", flush=True)
    packed = pack(exs, max_seq_len=max_seq_len, pad_id=PAD_ID)
    del exs
    # int16 arrays: ~14x less RAM than python lists at multi-B-token scale
    for d in packed:
        d["input_ids"] = np.asarray(d["input_ids"], dtype=np.int16)
        d["labels"] = np.asarray(d["labels"], dtype=np.int16)
    return packed
