"""BARC induction prompt construction (faithful to xu3kev/BARC gen_dataset_both.py).

Grids are serialized as COLOR NAMES (0->Black ... 9->Brown), cells space-separated,
rows newline-separated, with a trailing newline per grid. No transpose.
"""
from __future__ import annotations
from typing import List
from arc.grids import Grid

COLOR_NAMES = {0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
               5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"}

SYSTEM = ("You are a world-class puzzle solver with exceptional pattern recognition skills "
          "and expertise in Python programming. Your task is to analyze puzzles and provide "
          "Python solutions.")

_FINAL = ("\n\nWrite a Python function `transform` that can convert any given input grid to its "
          "corresponding output grid based on the pattern observed in the reference examples.\n")


def grid_to_str(grid: Grid) -> str:
    return "\n".join(" ".join(COLOR_NAMES[int(c)] for c in row) for row in grid) + "\n"


def build_messages(train_pairs: List[dict], test_input: Grid) -> List[dict]:
    q = ("Given input-output grid pairs as reference examples, carefully observe the patterns to "
         "predict the output grid for new test input. Each pair follows the same transformation "
         "rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces "
         "and rows by newlines.\n"
         "Here are the input and output grids for the reference examples:\n")
    for i, p in enumerate(train_pairs):
        q += f"Example {i+1}\nInput:\n{grid_to_str(p['input'])}Output:\n{grid_to_str(p['output'])}\n\n"
    q += f"Here is the input grid for the test example:\nInput:\n{grid_to_str(test_input)}"
    q += _FINAL
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
