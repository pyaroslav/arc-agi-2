"""Model bundle: load a base LLM, restrict decoding to the grid vocabulary,
attach a per-task LoRA, and build supervised (prompt, target) examples.

Design choice: instead of physically shrinking the embedding table to 16 tokens
(NVARC's throughput optimization, which requires remapping every token id), we
keep the base tokenizer and **constrain decoding** to the token ids that grid
content can use (digits, newline, im_end). Qwen tokenizers already split numbers
into single-digit tokens, so this reproduces the 16-token behavior for free and
is simpler/safer. Physical reduction can be added later purely for speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from .tokenizer import grid_to_str
from .grids import Grid

Message = Dict[str, str]


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    device: str
    allowed_ids: torch.Tensor   # grid-content token ids (digits + newline + im_end)
    digit_ids: List[int]
    nl_id: int
    im_end_id: int


def build_grid_token_ids(tok):
    digit_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(x) == 1 for x in digit_ids), "base tokenizer must split digits into single tokens"
    digit_ids = [x[0] for x in digit_ids]
    nl = tok.encode("\n", add_special_tokens=False)
    assert len(nl) == 1, "newline must be a single token"
    nl_id = nl[0]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    allowed = sorted(set(digit_ids) | {nl_id, im_end_id})
    return digit_ids, nl_id, im_end_id, allowed


def load_base(base_model: str, dtype: str = "bfloat16", attn_implementation: str = "sdpa",
              device: str = "cuda") -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(base_model)
    digit_ids, nl_id, im_end_id, allowed = build_grid_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=getattr(torch, dtype), attn_implementation=attn_implementation,
    ).to(device)
    return ModelBundle(model=model, tokenizer=tok, device=device,
                       allowed_ids=torch.tensor(allowed, device=device),
                       digit_ids=digit_ids, nl_id=nl_id, im_end_id=im_end_id)


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """Mask generation to a fixed set of token ids (the grid vocabulary)."""

    def __init__(self, allowed_ids: torch.Tensor):
        self.allowed_ids = allowed_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_ids.to(scores.device)] = 0.0
        return scores + mask


def add_lora(bundle: ModelBundle, r: int, alpha: int, dropout: float,
             target_modules: List[str]) -> ModelBundle:
    """Attach a fresh LoRA adapter (used for per-task TTFT). Returns the bundle
    with ``bundle.model`` replaced by the PEFT-wrapped model."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=target_modules, task_type="CAUSAL_LM", bias="none")
    bundle.model = get_peft_model(bundle.model, cfg)
    return bundle


# --------------------------------------------------------------------------- #
# Tokenizing prompts / supervised examples
# --------------------------------------------------------------------------- #
def _render_context(pairs: List[dict]) -> List[Message]:
    msgs: List[Message] = []
    for p in pairs:
        msgs.append({"role": "user", "content": grid_to_str(p["input"])})
        msgs.append({"role": "assistant", "content": grid_to_str(p["output"])})
    return msgs


def _chat_text(tok, msgs: List[Message]) -> str:
    """Render chat messages to text. Disable 'thinking' mode where supported
    (Qwen3) so the model emits the grid directly rather than a reasoning block."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_ids(bundle: ModelBundle, context_pairs: List[dict], test_input: Grid) -> torch.Tensor:
    """Token ids for [demos..., user(test_input), <assistant gen prompt>]."""
    tok = bundle.tokenizer
    msgs = _render_context(context_pairs) + [{"role": "user", "content": grid_to_str(test_input)}]
    text = _chat_text(tok, msgs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids], device=bundle.device)


def build_supervised_example(bundle: ModelBundle, context_pairs: List[dict],
                             query_input: Grid, query_output: Grid):
    """Return (input_ids, labels) where labels are -100 except over the target
    output grid tokens (+ trailing im_end). Built by concatenating a tokenized
    prompt with a separately tokenized target, so masking needs no offset maths.
    """
    tok = bundle.tokenizer
    prompt_ids = build_prompt_ids(bundle, context_pairs, query_input)[0]
    target_str = grid_to_str(query_output)
    target_ids = tok.encode(target_str, add_special_tokens=False) + [bundle.im_end_id]
    target_ids = torch.tensor(target_ids, device=bundle.device)
    input_ids = torch.cat([prompt_ids, target_ids])
    labels = torch.cat([
        torch.full((len(prompt_ids),), -100, device=bundle.device, dtype=torch.long),
        target_ids,
    ])
    return input_ids, labels


@torch.no_grad()
def sequence_logprob(bundle: ModelBundle, context_pairs: List[dict],
                     query_input: Grid, candidate_output: Grid) -> float:
    """Mean per-token log-prob the model assigns to ``candidate_output`` given the
    demos + query input. Used by the voting re-scorer."""
    input_ids, labels = build_supervised_example(bundle, context_pairs, query_input, candidate_output)
    out = bundle.model(input_ids=input_ids.unsqueeze(0))
    logits = out.logits[0, :-1]
    tgt = labels[1:]
    mask = tgt != -100
    if mask.sum() == 0:
        return float("-inf")
    logp = torch.log_softmax(logits[mask].float(), dim=-1)
    chosen = logp[torch.arange(mask.sum()), tgt[mask]]
    return chosen.mean().item()
