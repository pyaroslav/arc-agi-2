"""Build an SFT corpus in the NVARC 16-token grid format.

Each training example = a task's demonstration pairs rendered as alternating
user(input)/assistant(output) turns; loss is masked to the OUTPUT grids only
(completion-only), so the model learns the input->output mapping. We augment
(D4 × color-permutation × example order/subset) and sequence-pack short examples
to fill `max_seq_len`. Sources: ARC-AGI-2 train, NVARC synthetic puzzles, ReARC,
ConceptARC — anything in {"train":[{input,output}...]} or a flat list of pairs.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Dict, Iterator
import numpy as np

from nvarc.arc_loader import QwenFormatter
from nvarc.solver_local import ArcTokenizer, build_completion_labels
from arc.data import sample_augmentations


def load_any_tasks(path: str) -> Iterator[Dict]:
    """Yield {'pairs': [{input,output}...]} from a dir/file of ARC-like JSON."""
    files = ([path] if path.endswith(".json") else
             sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        pairs = None
        if isinstance(d, dict) and "train" in d:
            pairs = list(d["train"]) + [t for t in d.get("test", []) if "output" in t]
        elif isinstance(d, list) and d and isinstance(d[0], dict) and "input" in d[0]:
            pairs = d
        elif isinstance(d, dict) and "examples" in d:
            pairs = d["examples"]
        if pairs:
            pairs = [p for p in pairs if "input" in p and "output" in p]
            if len(pairs) >= 2:
                yield {"pairs": pairs}


def make_example(tok: ArcTokenizer, fmt: QwenFormatter, pairs: List[dict]):
    """(input_ids, labels) for one rendered task; labels masked to outputs."""
    text = fmt.fmt_train(pairs)
    ids = tok.encode(text)
    labels = build_completion_labels(ids)
    return ids, labels


def iter_examples(sources: List[str], *, n_aug: int = 4, pairs_per_example: int = 4,
                  max_seq_len: int = 8192, fix_background: bool = False, seed: int = 0,
                  limit_tasks: int = None) -> Iterator[tuple]:
    """Yield (input_ids, labels) SFT examples from the given source paths."""
    tok = ArcTokenizer()
    fmt = QwenFormatter(tok)
    rng = np.random.default_rng(seed)
    n_tasks = 0
    for src in sources:
        for task in load_any_tasks(src):
            if limit_tasks and n_tasks >= limit_tasks:
                return
            n_tasks += 1
            pool = task["pairs"]
            augs = sample_augmentations(n_aug, rng, n_demos=min(len(pool), pairs_per_example),
                                        include_identity=True, fix_background=fix_background)
            for aug in augs:
                k = min(len(pool), pairs_per_example)
                idx = rng.permutation(len(pool))[:k]
                chosen = [pool[i] for i in idx]
                a_pairs = [{"input": aug.apply_grid(p["input"]),
                            "output": aug.apply_grid(p["output"])} for p in chosen]
                ids, labels = make_example(tok, fmt, a_pairs)
                if 0 < len(ids) <= max_seq_len and any(l != -100 for l in labels):
                    yield ids, labels


def pack(examples: Iterator[tuple], max_seq_len: int = 8192, pad_id: int = 13):
    """Greedily pack (ids,labels) into <=max_seq_len sequences. Returns list of dicts."""
    packed, cur_ids, cur_lab = [], [], []
    for ids, labels in examples:
        if len(cur_ids) + len(ids) > max_seq_len and cur_ids:
            packed.append({"input_ids": cur_ids, "labels": cur_lab})
            cur_ids, cur_lab = [], []
        cur_ids += ids
        cur_lab += labels
    if cur_ids:
        packed.append({"input_ids": cur_ids, "labels": cur_lab})
    return packed
