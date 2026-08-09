"""Task loading + invertible task augmentation.

Augmentation = (D4 geometry) x (color permutation) x (demonstration-pair order).
The same augmentation that is applied to a task to produce a "view" can be
*inverted* on a predicted test output to recover a prediction in the original
frame. This is what makes augment-and-vote inference work.
"""
from __future__ import annotations

import json
import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from . import grids
from .grids import Grid


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
Task = Dict[str, list]  # {"train": [{"input","output"}...], "test": [{"input", ["output"]}...]}


def load_task(path: str) -> Task:
    with open(path) as f:
        return json.load(f)


def load_task_dir(directory: str) -> Dict[str, Task]:
    """Load all ``*.json`` tasks in a directory keyed by task id (filename stem)."""
    tasks: Dict[str, Task] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        tid = os.path.splitext(os.path.basename(path))[0]
        tasks[tid] = load_task(path)
    return tasks


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Augmentation:
    flip: int = 0                       # D4: transpose flag
    rot: int = 0                        # D4: number of 90deg CCW rotations
    color_perm: tuple = field(default_factory=lambda: tuple(range(grids.NUM_COLORS)))
    example_perm: Optional[tuple] = None  # reorder demo pairs; None = keep order

    # ---- grid-level ops ----
    def apply_grid(self, g: Grid) -> Grid:
        g = grids.apply_color(g, list(self.color_perm))
        g = grids.apply_d4(g, self.flip, self.rot)
        return g

    def invert_grid(self, g: Grid) -> Grid:
        """Map a grid produced in this augmented frame back to the original frame."""
        g = grids.invert_d4(g, self.flip, self.rot)
        g = grids.invert_color(g, list(self.color_perm))
        return g

    # ---- task-level ----
    def apply_task(self, task: Task) -> Task:
        train = [{"input": self.apply_grid(p["input"]),
                  "output": self.apply_grid(p["output"])} for p in task["train"]]
        if self.example_perm is not None:
            train = [train[i] for i in self.example_perm]
        test = [{"input": self.apply_grid(t["input"])} for t in task["test"]]
        return {"train": train, "test": test}


IDENTITY = Augmentation()


def sample_augmentations(
    n: int,
    rng: np.random.Generator,
    n_demos: int,
    include_identity: bool = True,
    fix_background: bool = False,
    permute_examples: bool = True,
) -> List[Augmentation]:
    """Sample ``n`` distinct-ish augmentations. First is identity if requested."""
    augs: List[Augmentation] = []
    seen = set()
    if include_identity:
        augs.append(IDENTITY)
        seen.add((0, 0, IDENTITY.color_perm, None))
    guard = 0
    while len(augs) < n and guard < n * 50:
        guard += 1
        flip = int(rng.integers(0, 2))
        rot = int(rng.integers(0, 4))
        cperm = tuple(grids.random_color_perm(rng, fix_background=fix_background))
        eperm = tuple(rng.permutation(n_demos).tolist()) if permute_examples else None
        key = (flip, rot, cperm, eperm)
        if key in seen:
            continue
        seen.add(key)
        augs.append(Augmentation(flip=flip, rot=rot, color_perm=cperm, example_perm=eperm))
    return augs
