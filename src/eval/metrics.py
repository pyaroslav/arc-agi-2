"""Competition scoring: pass@2.

For each test output the solver submits up to 2 predicted grids; it is correct
if *either* exactly matches the ground truth. The task/overall score is the mean
over all test outputs.
"""
from __future__ import annotations

from typing import List, Dict
from arc.grids import Grid, grids_equal


def output_correct(predictions: List[Grid], truth: Grid) -> bool:
    """True if any of the (<=2) predictions exactly equals the ground truth."""
    return any(grids_equal(p, truth) for p in predictions[:2])


def score_task(predictions_per_output: List[List[Grid]], truths: List[Grid]) -> float:
    """Mean pass@2 over the test outputs of a single task."""
    assert len(predictions_per_output) == len(truths)
    if not truths:
        return 0.0
    hits = sum(output_correct(preds, t) for preds, t in zip(predictions_per_output, truths))
    return hits / len(truths)


def score_dataset(results: Dict[str, Dict]) -> Dict[str, float]:
    """Aggregate scores across tasks.

    ``results[task_id] = {"predictions": [[g1,g2], ...], "truths": [t, ...]}``
    Returns overall pass@2 averaged **per test output** (the competition metric)
    plus a per-task-averaged number for reference.
    """
    total_outputs = 0
    total_hits = 0
    per_task = []
    for r in results.values():
        preds, truths = r["predictions"], r["truths"]
        for p, t in zip(preds, truths):
            total_outputs += 1
            total_hits += int(output_correct(p, t))
        per_task.append(score_task(preds, truths))
    return {
        "pass@2_per_output": (total_hits / total_outputs) if total_outputs else 0.0,
        "pass@2_per_task": (sum(per_task) / len(per_task)) if per_task else 0.0,
        "n_tasks": len(results),
        "n_outputs": total_outputs,
    }
