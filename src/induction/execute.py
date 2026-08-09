"""Sandboxed execution + train-verification of generated `transform` programs.

Each program runs in a forked subprocess with a 2s timeout (kills infinite loops).
A program is kept only if it reproduces ALL train outputs exactly (np.array_equal);
survivors are applied to the test input(s). Output must be a 2-D int grid in 0-9.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
from typing import List, Optional

import numpy as np

from arc.grids import Grid

_CTX = mp.get_context("fork")
_DIR = os.path.dirname(__file__)  # holds common.py
_PREAMBLE = "from common import *\nimport numpy as np\nimport random\nrandom.seed(0)\nnp.random.seed(0)\n"


def parse_code(text: str) -> Optional[str]:
    """First ```python block (fallback: any ``` block)."""
    for pat in (r"```python(.*?)```", r"```(.*?)```"):
        m = re.findall(pat, text, re.DOTALL)
        if m:
            return m[0].strip()
    return None


def _run(code, inputs, q):
    import sys
    sys.path.insert(0, _DIR)
    try:
        ns = {}
        exec(_PREAMBLE + code, ns)
        fn = ns.get("transform") or ns.get("main")
        if fn is None:
            q.put([None] * len(inputs)); return
        outs = []
        for g in inputs:
            try:
                o = np.asarray(fn(np.array(g, dtype=int)))
                if (o.ndim == 2 and o.size > 0 and o.dtype.kind in "iu"
                        and int(o.min()) >= 0 and int(o.max()) <= 9):
                    outs.append(o.astype(int).tolist())
                else:
                    outs.append(None)
            except Exception:
                outs.append(None)
        q.put(outs)
    except Exception:
        q.put([None] * len(inputs))


def run_program(code: str, inputs: List[Grid], timeout: float = 2.0) -> List[Optional[Grid]]:
    q = _CTX.Queue()
    p = _CTX.Process(target=_run, args=(code, inputs, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return [None] * len(inputs)
    try:
        return q.get_nowait()
    except Exception:
        return [None] * len(inputs)


def verify_and_predict(code: str, train_pairs: List[dict], test_inputs: List[Grid],
                       timeout: float = 2.0):
    """Return (passes_all_train: bool, [test_output|None ...])."""
    inputs = [p["input"] for p in train_pairs] + list(test_inputs)
    outs = run_program(code, inputs, timeout)
    n = len(train_pairs)
    train_ok = all(outs[i] == train_pairs[i]["output"] for i in range(n))
    return train_ok, outs[n:]
