"""Candidate generation: sample output grids under several augmentations and
invert them back to the original frame. (A batched-DFS decoder — NVARC's speed
trick — can replace the sampler later; sampling is the correct, simple v1.)
"""
from __future__ import annotations

from collections import Counter
from typing import List, Dict, Tuple
import torch

from .model import ModelBundle, build_prompt_ids, AllowedTokensLogitsProcessor
from .tokenizer import str_to_grid
from .grids import Grid, is_valid_grid
from .data import Augmentation


def _key(g: Grid):
    return tuple(tuple(row) for row in g)


def _aug_context(train_pairs: List[dict], aug: Augmentation) -> List[dict]:
    return [{"input": aug.apply_grid(p["input"]), "output": aug.apply_grid(p["output"])}
            for p in train_pairs]


@torch.no_grad()
def generate_candidates(bundle: ModelBundle, train_pairs: List[dict], test_input: Grid,
                        gen_augs: List[Augmentation], samples_per_aug: int,
                        temperature: float, top_p: float, max_new_tokens: int
                        ) -> Dict[Tuple, Tuple[Grid, int]]:
    """Return {grid_key: (grid_in_original_frame, generation_count)}."""
    proc = [AllowedTokensLogitsProcessor(bundle.allowed_ids)]
    cand: Dict[Tuple, List] = {}
    model = bundle.model
    for aug in gen_augs:
        ctx = _aug_context(train_pairs, aug)
        a_test = aug.apply_grid(test_input)
        prompt = build_prompt_ids(bundle, ctx, a_test)
        attn = torch.ones_like(prompt)
        do_sample = temperature and temperature > 0
        out = model.generate(
            prompt, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=samples_per_aug if do_sample else 1,
            logits_processor=proc, eos_token_id=bundle.im_end_id,
            pad_token_id=bundle.im_end_id,
        )
        plen = prompt.shape[1]
        for seq in out:
            text = bundle.tokenizer.decode(seq[plen:], skip_special_tokens=True)
            g = str_to_grid(text)
            if not is_valid_grid(g):
                continue
            g0 = aug.invert_grid(g)          # back to the original frame
            if not is_valid_grid(g0):
                continue
            k = _key(g0)
            if k not in cand:
                cand[k] = [g0, 0]
            cand[k][1] += 1
    return {k: (v[0], v[1]) for k, v in cand.items()}
