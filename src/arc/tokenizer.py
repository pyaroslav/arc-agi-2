"""Grid <-> text serialization and task -> chat-message formatting.

Mirrors the NVARC representation: a grid is serialized **row by row as digit
strings joined by newlines**; an ARC task becomes a chat dialog where each
demonstration pair is a (user=input grid, assistant=output grid) turn, and the
final user turn is the test input the model must complete.

The actual token ids come from the base model's HF tokenizer at train/infer
time. NVARC additionally prunes the embedding table to the **16 token strings**
that can ever appear (:func:`reduced_token_strings`); that model-build step lives
in the SFT/model code, not here. This module is pure string<->grid logic so it
is testable with no model or GPU.
"""
from __future__ import annotations

from typing import List, Dict
from .grids import Grid

# The 16 token strings that can appear once the vocabulary is reduced:
#   10 digit/color tokens, a newline, and the chat special tokens.
DIGIT_TOKENS = [str(d) for d in range(10)]
NEWLINE_TOKEN = "\n"
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
# (user/assistant role words are part of the chat template; NVARC counts the
#  start/end markers + pad. We expose the canonical printable set used by grids.)


def reduced_token_strings() -> List[str]:
    """The minimal set of literal token strings that grid dialogs require."""
    return DIGIT_TOKENS + [NEWLINE_TOKEN] + SPECIAL_TOKENS


# --------------------------------------------------------------------------- #
# Grid <-> string
# --------------------------------------------------------------------------- #
def grid_to_str(grid: Grid) -> str:
    """Rows of concatenated digits, joined by newlines. E.g. [[1,2],[3,4]] -> '12\\n34'."""
    return "\n".join("".join(str(c) for c in row) for row in grid)


def str_to_grid(s: str) -> Grid:
    """Parse a digit-grid string back to ``list[list[int]]``.

    Robust to trailing whitespace, blank lines, and stray non-digit chars
    (model output can be imperfect). Rows that contain no digits are dropped.
    Returns ``[]`` if nothing parseable.
    """
    rows: Grid = []
    for line in s.replace("\r", "").split("\n"):
        digits = [int(ch) for ch in line if ch.isdigit()]
        if digits:
            rows.append(digits)
    return rows


# --------------------------------------------------------------------------- #
# Task -> chat messages
# --------------------------------------------------------------------------- #
Message = Dict[str, str]


def task_to_messages(train_pairs: List[dict], test_input: Grid) -> List[Message]:
    """Build the chat dialog for a single test input.

    ``train_pairs`` is a list of ``{"input": Grid, "output": Grid}``. The result
    is [user(in), assistant(out), ... , user(test_in)] — the model continues
    with the assistant turn (the predicted test output grid).
    """
    messages: List[Message] = []
    for pair in train_pairs:
        messages.append({"role": "user", "content": grid_to_str(pair["input"])})
        messages.append({"role": "assistant", "content": grid_to_str(pair["output"])})
    messages.append({"role": "user", "content": grid_to_str(test_input)})
    return messages


def fallback_chat_text(messages: List[Message], add_generation_prompt: bool = True) -> str:
    """Render messages with a Qwen-style template *without* an HF tokenizer.

    Used only for offline tests / debugging; production tokenization uses the
    model's own ``apply_chat_template``.
    """
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    text = "\n".join(parts)
    if add_generation_prompt:
        text += "\n<|im_start|>assistant\n"
    return text
