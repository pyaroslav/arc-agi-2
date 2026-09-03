"""Adapter: sorokin's arrow corpus (puzzle_name, messages[]) -> packed SFT data.

Renders chat turns in the exact QwenFormatter template, labels completion-only
(assistant spans), and stores EVERYTHING as int16 numpy end-to-end so a
multi-B-token build stays ~2 bytes/token in RAM (python lists are ~28).
"""
from __future__ import annotations

import glob
import random

import numpy as np
import pyarrow as pa

from nvarc.solver_local import ArcTokenizer, build_completion_labels

PAD_ID = 13


def render(messages) -> str:
    out = []
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        out.append("<|im_start|>" + role + "\n" + m["content"] + "<|im_end|>")
    return "".join(out)


def build_arrow(sources, max_rows, max_seq_len, seed=0):
    tok = ArcTokenizer()
    files = []
    for s in sources:
        files += sorted(glob.glob(s.rstrip("/") + "/data-*.arrow"))
    rng = random.Random(seed)
    rng.shuffle(files)
    packed = []
    cur_ids, cur_lab = [], []

    def flush():
        nonlocal cur_ids, cur_lab
        if cur_ids:
            packed.append({"input_ids": np.asarray(cur_ids, dtype=np.int16),
                           "labels": np.asarray(cur_lab, dtype=np.int16)})
            cur_ids, cur_lab = [], []

    n_seen = n_used = 0
    for f in files:
        table = pa.ipc.open_stream(open(f, "rb")).read_all()
        idx = list(range(table.num_rows))
        rng.shuffle(idx)
        rows = table.column("messages").to_pylist()
        for i in idx:
            if max_rows and n_seen >= max_rows:
                break
            n_seen += 1
            try:
                ids = tok.encode(render(rows[i]))
            except Exception:
                continue
            if len(ids) > max_seq_len:
                continue
            labels = build_completion_labels(ids)
            if all(l == -100 for l in labels):
                continue
            n_used += 1
            if len(cur_ids) + len(ids) > max_seq_len and cur_ids:
                flush()
            cur_ids += ids
            cur_lab += labels
            if n_used % 50000 == 0:
                print(f"  arrow build: {n_used} examples, {len(packed)} packed", flush=True)
        del table, rows
        if max_rows and n_seen >= max_rows:
            break
    flush()
    rng.shuffle(packed)
    print(f"arrow corpus: {n_seen} rows seen, {n_used} usable -> {len(packed)} packed", flush=True)
    return packed


def save_packed(packed, path):
    ids = np.concatenate([d["input_ids"] for d in packed])
    lab = np.concatenate([d["labels"] for d in packed])
    lens = np.asarray([len(d["input_ids"]) for d in packed], dtype=np.int32)
    np.savez(path, ids=ids, labels=lab, lengths=lens)
    print(f"saved packed cache: {len(packed)} seqs -> {path}", flush=True)


def load_packed(path):
    z = np.load(path, mmap_mode="r")
    ids, lab, lens = z["ids"], z["labels"], z["lengths"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    packed = [{"input_ids": ids[offs[i]:offs[i+1]], "labels": lab[offs[i]:offs[i+1]]}
              for i in range(len(lens))]
    print(f"loaded packed cache: {len(packed)} seqs from {path}", flush=True)
    return packed
