#!/usr/bin/env python3
"""Refresh docs/lb_draws.csv from the full Kaggle submission history.

Usage: python scripts/pull_lb_draws.py            (writes docs/lb_draws.csv)
Needs the Kaggle CLI (>= 2.2) configured; the plain `submissions` call caps at
50 rows, so --page-size 200 is required for the full series.
"""
import csv, io, pathlib, subprocess, sys
COMP = "arc-prize-2026-arc-agi-2"
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "lb_draws.csv"
raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "200"],
                     capture_output=True, text=True, check=True).stdout
lines = [l for l in raw.splitlines() if not l.startswith("Warning")]
rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
rows.sort(key=lambda r: r["date"])
with OUT.open("w", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["date", "submission_id", "status", "public_score", "description"])
    for r in rows:
        w.writerow([r["date"][:19], r["ref"], r["status"].split(".")[-1], r["publicScore"], r["description"]])
print(f"{len(rows)} submissions -> {OUT}")
