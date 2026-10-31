"""Shared logging/summary helpers. The Isaac Lab script writes the same CSV columns
so mock, simulator and (later) real-robot results can be compared directly."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

COLUMNS = [
    "env", "task", "episode", "seed", "success", "termination", "steps", "local_repairs",
    "full_replans", "stale_detections", "collisions", "guard_violations", "goal_xy_err_mm",
    "decision_latency_ms",
]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def write_csv(rows: List[Dict], path: Path) -> None:
    extra = sorted({k for r in rows for k in r} - set(COLUMNS))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + extra)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(rows: Iterable[Dict]) -> Dict[str, Dict]:
    by_task: Dict[str, List[Dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    out = {}
    for task, rs in by_task.items():
        k, n = sum(bool(r["success"]) for r in rs), len(rs)
        lo, hi = wilson(k, n)
        mean = lambda key: float(np.nanmean([float(r[key]) for r in rs]))  # noqa: E731
        out[task] = {
            "episodes": n,
            "success_rate": k / n,
            "success_ci95": [round(lo, 3), round(hi, 3)],
            "mean_steps": mean("steps"),
            "mean_local_repairs": mean("local_repairs"),
            "mean_full_replans": mean("full_replans"),
            "mean_stale_detections": mean("stale_detections"),
            "collisions_per_ep": mean("collisions"),
            "guard_violations_per_ep": mean("guard_violations"),
            "mean_goal_xy_err_mm": mean("goal_xy_err_mm"),
            "mean_decision_latency_ms": mean("decision_latency_ms"),
            "terminations": {t: sum(r["termination"] == t for r in rs) for t in {r["termination"] for r in rs}},
        }
    return out


def print_table(summary: Dict[str, Dict]) -> None:
    hdr = f"{'task':<12}{'succ':>7}{'95% CI':>15}{'steps':>7}{'local':>7}{'full':>6}{'stale':>7}{'coll':>6}{'guard':>7}{'err mm':>8}{'dec ms':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t, s in summary.items():
        ci = f"[{s['success_ci95'][0]:.2f},{s['success_ci95'][1]:.2f}]"
        print(f"{t:<12}{s['success_rate']:>7.2f}{ci:>15}{s['mean_steps']:>7.1f}{s['mean_local_repairs']:>7.2f}"
              f"{s['mean_full_replans']:>6.2f}{s['mean_stale_detections']:>7.2f}{s['collisions_per_ep']:>6.2f}"
              f"{s['guard_violations_per_ep']:>7.2f}{s['mean_goal_xy_err_mm']:>8.1f}{s['mean_decision_latency_ms']:>8.3f}")


def write_summary(summary: Dict, path: Path, meta: Dict) -> None:
    with open(path, "w") as f:
        json.dump({"meta": meta, "tasks": summary}, f, indent=2)
