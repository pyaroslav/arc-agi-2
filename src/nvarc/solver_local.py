"""Single-GPU (RTX 5090 / Blackwell) port of NVARC's per-task solver.

Replaces their `unsloth.FastLanguageModel` + `UnslothTrainer` with
`transformers` + `peft` (validated on Blackwell; avoids unsloth's torch pins).
`turbo_dfs` / `calc_scores` are vendored ~verbatim — they only call `model(...)`
and `tokenizer.encode`, so they work unchanged.

CRITICAL: the checkpoint's WordLevel tokenizer maps `user`->11 / `assistant`->12,
but transformers>=5 drops those words on `.encode`. We therefore tokenize with our
own exact `ArcTokenizer` so the model sees the token stream it was trained on
(verified: with 11/12 the SFT model solves tasks; without, it solves none).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

sys.setrecursionlimit(20000)  # turbo_dfs recurses once per generated token (grid can be ~930 deep)
from typing import List
import numpy as np
import torch
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from .arc_loader import ArcDataset, QwenFormatter

# ---- 16-token vocabulary (from the checkpoint's vocab.json / added_tokens.json) ----
NEWLINE_ID = 10
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
IM_START_ID = 14
EOS_ID = 15
ARC_TOKENS = list(range(10)) + [NEWLINE_ID, EOS_ID]   # tokens the DFS may emit
_SPECIALS = {"<|im_start|>": IM_START_ID, "<|im_end|>": EOS_ID, "<|endoftext|>": PAD_ID}
_DECODE = {**{i: str(i) for i in range(10)}, NEWLINE_ID: "\n",
           USER_TOKEN_ID: "user", ASSISTANT_TOKEN_ID: "assistant",
           PAD_ID: "<|endoftext|>", IM_START_ID: "<|im_start|>", EOS_ID: "<|im_end|>"}


class ArcTokenizer:
    """Exact encoder/decoder for the checkpoint's WordLevel 16-token vocab."""
    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    bos_token_id = PAD_ID

    def encode(self, text, **kw) -> List[int]:
        ids, i, n = [], 0, len(text)
        while i < n:
            matched = False
            for s, t in _SPECIALS.items():
                if text.startswith(s, i):
                    ids.append(t); i += len(s); matched = True; break
            if matched:
                continue
            if text.startswith("user", i):
                ids.append(USER_TOKEN_ID); i += 4; continue
            if text.startswith("assistant", i):
                ids.append(ASSISTANT_TOKEN_ID); i += 9; continue
            ch = text[i]
            if ch == "\n":
                ids.append(NEWLINE_ID)
            elif ch.isdigit():
                ids.append(int(ch))
            i += 1
        return ids

    def decode(self, ids, **kw) -> str:
        return "".join(_DECODE.get(int(t), "") for t in ids)

    def __len__(self):
        return 16


def build_completion_labels(ids: List[int]) -> List[int]:
    """Completion-only masking (NVARC collator): train only on assistant output
    grids. Even segments = user/input (masked), odd = assistant/output (kept)."""
    arr = np.asarray(ids)
    starts = sorted(np.where(arr == USER_TOKEN_ID)[0].tolist() +
                    np.where(arr == ASSISTANT_TOKEN_ID)[0].tolist())
    ends = np.where(arr == EOS_ID)[0].tolist()
    labels = [-100] * len(ids)
    for j, (s, e) in enumerate(zip(starts, ends)):
        if j % 2 == 1 and s < e:           # assistant turn
            s2, e2 = s + 2, e + 1          # skip 'assistant','\n'; include trailing eos
            for k in range(s2, min(e2, len(ids))):
                labels[k] = ids[k]
    return labels


# --------------------------------------------------------------------------- #
# Decoder + scorer (vendored ~verbatim from NVARC arc_solver.py)
# --------------------------------------------------------------------------- #
def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache,
              start_time, end_time, time_budget=540) -> dict:
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    suffixes = defaultdict(list)
    candidates = dict()
    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))
    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x: x[0])
    while time.time() - start_time < time_budget and time.time() < end_time:
        batch_tokens, batch_scores, num_alive = [], [], 0
        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID); batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t); batch_scores.append(score); num_alive += 1
        if num_alive == 0:
            break
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache, return_dict=True, use_cache=True,
        )
        next_suffixes = turbo_dfs(model, outputs.logits[:, -1], max_new_tokens - 1, max_score,
                                  batch_scores, pos + 1, outputs.past_key_values,
                                  start_time, end_time, time_budget)
        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))
    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, time_budget=540):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(model, outputs.logits[:, -1], max_new_tokens, max_score,
                         [0.0] * input_ids.size(0), input_ids.size(1), outputs.past_key_values,
                         time.time(), end_time, time_budget)
    return [(bid, sorted(beams, key=lambda x: x[0])) for bid, beams in suffixes.items()]


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    bqt, bat, bt, bl = [], [], [], []
    for query, answer in zip(queries, answers):
        qt = tokenizer.encode(query); at = tokenizer.encode(answer)
        bqt.append(qt); bat.append(at); bt.append(qt + at); bl.append(len(qt + at))
    max_len = max(bl)
    padded = [t + [PAD_ID] * (max_len - len(t)) for t in bt]
    input_ids = torch.tensor(padded, device=model.device, dtype=torch.long)
    logits = model(input_ids=input_ids, return_dict=True, use_cache=True).logits.float().cpu().log_softmax(-1)
    out = []
    for lg, qt, at in zip(logits, bqt, bat):
        ql = len(qt)
        al = lg[ql - 1:ql - 1 + len(at)]
        out.append(-al[torch.arange(len(at)), at].sum().item())
    return out


# --------------------------------------------------------------------------- #
# Per-task solver
# --------------------------------------------------------------------------- #
class LocalSolver:
    def __init__(self, model_dir: str, device: str = "cuda", max_seq_len: int = 8192,
                 lora_r: int = 256, lora_alpha: int = 32, dfs_prob: float = 0.2):
        self.device = device
        self.max_seq_len = max_seq_len
        self.tokenizer = ArcTokenizer()
        self.formatter = QwenFormatter(self.tokenizer)
        self.max_new_tokens = self.formatter.max_new_tokens()
        # turbo_dfs keeps token paths with per-token prob > dfs_prob (lower => more
        # candidates => higher coverage, at more DFS compute).
        self.max_score = -np.log(dfs_prob)
        base = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj",
                                         "embed_tokens", "lm_head"],
                         use_rslora=True, task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)
        for _, p in self.model.named_parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
        self.init_state = {k: v.clone().detach()
                           for k, v in get_peft_model_state_dict(self.model).items()}

    def reset_lora(self):
        set_peft_model_state_dict(self.model, {k: v.clone() for k, v in self.init_state.items()})

    def train_task(self, puzzle_ds, *, aug_n=16, lr=5e-5, seed=1, log=False, grad_ckpt=True,
                   epochs=1):
        self.reset_lora()
        self.model.train()
        if grad_ckpt:
            # Fit 8192-token TTFT in ~30GB (no unsloth). Must disable KV cache while on.
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False
        train_ds = puzzle_ds.augment(n=aug_n, shfl_keys=True, seed=seed)
        train_ds = train_ds.cut_to_len(self.formatter, name="text", max_len=self.max_seq_len)
        examples = [self.tokenizer.encode(d["text"]) for d in train_ds.as_list(self.formatter)]
        examples = [e for e in examples if len(e) <= self.max_seq_len]
        rng = np.random.default_rng(seed)
        rng.shuffle(examples)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        total = max(1, len(examples) * epochs)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
        run = []
        for ep in range(epochs):
            order = list(examples)
            np.random.default_rng(seed + ep).shuffle(order)
            run.extend(order)
        for step, ids in enumerate(run):
            labels = build_completion_labels(ids)
            if all(l == -100 for l in labels):
                continue
            inp = torch.tensor([ids], device=self.device)
            lab = torch.tensor([labels], device=self.device)
            loss = self.model(input_ids=inp, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if log and step % 32 == 0:
                print(f"    ttft {step}/{total} loss={loss.item():.4f}")
        if grad_ckpt:
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True          # restore for DFS inference
        self.model.eval()

    @torch.no_grad()
    def solve_task(self, key, arc_test_set, *, eval_aug_n=2, batch_size=4, end_time=None,
                   time_budget=540):
        """Return decoded_results for one task: {base_key: {subkey.outN: sample}}."""
        if end_time is None:
            end_time = time.time() + 3600
        fmt = self.formatter
        puzzle_ds = arc_test_set.change_keys([key])
        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        eval_ds = puzzle_ds_multi.augment(n=eval_aug_n, seed=2)
        eval_ds = eval_ds.cut_to_len(fmt, name="input", max_len=self.max_seq_len - self.max_new_tokens)

        by_test = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            by_test[subkey.split(".")[0]].append(subkey)

        decoded_results = {}
        known_scores = {}
        for base_key, subkeys in by_test.items():
            for i in range(0, len(subkeys), batch_size):
                if time.time() > end_time:
                    break
                batch = subkeys[i:i + batch_size]
                prefixes = [self.tokenizer.encode(eval_ds.get(sk, fmt)["input"]) for sk in batch]
                # pad-left not needed: DFS handles per-row; run equal-length via separate calls if ragged
                maxl = max(len(p) for p in prefixes)
                if any(len(p) != maxl for p in prefixes):
                    # run each separately to avoid mixing positions (simple + correct)
                    results = []
                    for sk, pfx in zip(batch, prefixes):
                        results += [(sk, b) for _, b in inference_turbo_dfs(
                            self.model, [pfx], self.max_new_tokens, self.max_score, end_time, time_budget)]
                else:
                    raw = inference_turbo_dfs(self.model, prefixes, self.max_new_tokens,
                                              self.max_score, end_time, time_budget)
                    results = [(batch[bid], beams) for bid, beams in raw]
                for subkey, beams in results:
                    bk = subkey.split(".")[0]
                    decoded_results.setdefault(bk, {})
                    samples = []
                    for beam_score, toks in beams:
                        array = fmt.convert_tokens_to_array(toks)
                        if array is None:
                            continue
                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        gid = (bk, tuple(map(tuple, solution)))
                        if gid in known_scores:
                            aug_scores = known_scores[gid]
                        else:
                            aug_ds = ArcDataset(keys=[bk], queries={bk: puzzle_ds_multi.queries.get(bk)},
                                                replies={bk: [solution.tolist()]})
                            aug_ds = aug_ds.augment(seed=hash(bk) % 1024 ** 2)
                            aug_ds = aug_ds.cut_to_len(fmt, name="input", max_len=self.max_seq_len - self.max_new_tokens)
                            aq, aa = [], []
                            for s in aug_ds.as_list(fmt):
                                aq.append(s["input"]); aa.append(s["reply"])
                            aug_scores = calc_scores(aq[:4], aa[:4], self.tokenizer, self.model) + \
                                         calc_scores(aq[4:], aa[4:], self.tokenizer, self.model)
                            known_scores[gid] = aug_scores
                        samples.append({"beam_score": beam_score, "score_aug": aug_scores,
                                        "solution": solution})
                    for j, s in enumerate(samples):
                        decoded_results[bk][f"{subkey}.out{j}"] = s
        return decoded_results
