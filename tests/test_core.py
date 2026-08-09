"""Unit tests for the pure-Python core: serialization, D4/color invertibility,
full task-augmentation round-trip, and the pass@2 metric.

Run:  cd src && python -m pytest ../tests/test_core.py -q
(the repo's conftest puts ``src`` on sys.path)
"""
import glob
import json
import os

import numpy as np
import pytest

from arc import grids, data
from arc.tokenizer import grid_to_str, str_to_grid, task_to_messages
from arc.data import Augmentation, sample_augmentations
from eval.metrics import output_correct, score_task, score_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "arc-agi-2")


def _rand_grid(rng, hmax=8, wmax=8):
    h = int(rng.integers(1, hmax + 1))
    w = int(rng.integers(1, wmax + 1))
    return [[int(rng.integers(0, 10)) for _ in range(w)] for _ in range(h)]


# --------------------------------------------------------------------------- #
def test_serialization_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = _rand_grid(rng)
        assert str_to_grid(grid_to_str(g)) == g


def test_str_to_grid_robust_to_noise():
    assert str_to_grid("12\n34\n") == [[1, 2], [3, 4]]
    assert str_to_grid("  12 \n\n 34 ") == [[1, 2], [3, 4]]
    assert str_to_grid("garbage\n12\n--") == [[1, 2]]
    assert str_to_grid("") == []


def test_d4_is_group_of_8_and_invertible():
    rng = np.random.default_rng(1)
    g = _rand_grid(rng, 6, 5)  # non-square, asymmetric
    seen = set()
    for flip, rot in grids.D4_ELEMENTS:
        t = grids.apply_d4(g, flip, rot)
        assert grids.invert_d4(t, flip, rot) == g, (flip, rot)
        seen.add(tuple(map(tuple, t)))
    # 8 distinct symmetries for a generic asymmetric grid
    assert len(seen) == 8


def test_color_perm_invertible():
    rng = np.random.default_rng(2)
    g = _rand_grid(rng)
    for fix_bg in (False, True):
        perm = grids.random_color_perm(rng, fix_background=fix_bg)
        assert grids.invert_color(grids.apply_color(g, perm), perm) == g
        if fix_bg:
            assert perm[0] == 0


def test_augmentation_output_roundtrip():
    """A predicted output in an augmented frame inverts back to the original."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        out = _rand_grid(rng)
        flip = int(rng.integers(0, 2)); rot = int(rng.integers(0, 4))
        perm = tuple(grids.random_color_perm(rng))
        aug = Augmentation(flip=flip, rot=rot, color_perm=perm)
        assert aug.invert_grid(aug.apply_grid(out)) == out


def test_apply_task_shapes_and_example_perm():
    task = {
        "train": [{"input": [[1]], "output": [[2]]},
                  {"input": [[3]], "output": [[4]]},
                  {"input": [[5]], "output": [[6]]}],
        "test": [{"input": [[7]], "output": [[8]]}],
    }
    aug = Augmentation(example_perm=(2, 0, 1))
    t = aug.apply_task(task)
    assert [p["input"][0][0] for p in t["train"]] == [5, 1, 3]
    assert "output" not in t["test"][0]  # test outputs are stripped from the view


def test_sample_augmentations_distinct():
    rng = np.random.default_rng(4)
    augs = sample_augmentations(8, rng, n_demos=3)
    assert augs[0] == data.IDENTITY
    assert len(augs) == 8


def test_pass_at_2_metric():
    truth = [[1, 2], [3, 4]]
    wrong = [[0, 0], [0, 0]]
    assert output_correct([wrong, truth], truth) is True
    assert output_correct([truth, wrong], truth) is True
    assert output_correct([wrong, wrong], truth) is False
    assert output_correct([wrong, wrong, truth], truth) is False  # only first 2 count
    assert score_task([[truth, wrong], [wrong, wrong]], [truth, [[9]]]) == 0.5
    res = {"t": {"predictions": [[truth, wrong]], "truths": [truth]}}
    agg = score_dataset(res)
    assert agg["pass@2_per_output"] == 1.0 and agg["n_outputs"] == 1


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason="ARC-AGI-2 data not present")
def test_real_data_roundtrip_and_format():
    files = glob.glob(os.path.join(DATA_DIR, "training", "*.json"))
    assert len(files) == 1000
    rng = np.random.default_rng(5)
    n_checked = 0
    for path in [files[i] for i in rng.integers(0, len(files), 50)]:
        task = json.load(open(path))
        assert "train" in task and "test" in task
        for split in ("train", "test"):
            for pair in task[split]:
                g = pair["input"]
                assert grids.is_valid_grid(g)
                assert str_to_grid(grid_to_str(g)) == g  # serialization round-trip
                n_checked += 1
        # messages build without error
        msgs = task_to_messages(task["train"], task["test"][0]["input"])
        assert msgs[-1]["role"] == "user"
    assert n_checked > 0
