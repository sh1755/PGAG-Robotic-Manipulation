"""Evaluation harness.

    python -m eval.run                                   # all tasks, default config
    python -m eval.run --tasks stack --episodes 20
    python -m eval.run --config configs/stress.yaml --no-local-repair --tag full_replan

Writes runs/<timestamp>_<tag>/ with:
    episodes.csv   one row per episode (columns in eval/metrics.py)
    events.jsonl   every executor decision, for debugging a failed episode
    summary.json   per-task success rate with 95% Wilson CI and mean metrics
    config.yaml    the exact config used
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from envs.mock_env import MockTabletopEnv
from envs.scenes import SCENES
from execution.executor import ExecutorConfig, GrammarExecutor
from grammar.programs import TASKS

from .metrics import print_table, summarise, write_csv, write_summary


def make_env(cfg: dict, task: str):
    if cfg["env"] == "mock":
        return MockTabletopEnv(SCENES[task], cfg.get("mock", {}), xy_noise=cfg["randomisation"]["object_xy_noise"])
    raise ValueError(
        f"env '{cfg['env']}' is not wired into this harness yet. In Week 1 the simulator runs "
        "through sim/isaaclab_grasp.py, which writes the same CSV format."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval_default.yaml")
    ap.add_argument("--tasks", nargs="*", choices=list(TASKS))
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-local-repair", action="store_true", help="full-replanning baseline")
    ap.add_argument("--tag", default=None, help="label appended to the run folder name")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.tasks:
        cfg["tasks"] = args.tasks
    if args.episodes:
        cfg["episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_local_repair:
        cfg["executor"]["local_repair"] = False

    tag = args.tag or ("local" if cfg["executor"]["local_repair"] else "fullreplan")
    run_dir = Path(cfg.get("out_dir", "runs")) / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)

    rows = []
    with open(run_dir / "events.jsonl", "w") as ev_file:
        for task in cfg["tasks"]:
            program = TASKS[task]()
            problems = program.validate()
            if problems:
                raise SystemExit(f"Program '{task}' is invalid:\n  " + "\n  ".join(problems))
            env = make_env(cfg, task)
            for ep in range(cfg["episodes"]):
                seed = cfg["seed"] * 100_000 + ep
                env.reset(seed)
                res = GrammarExecutor(program, env, ExecutorConfig(**cfg["executor"])).run()
                rows.append({
                    "env": cfg["env"], "task": task, "episode": ep, "seed": seed,
                    "success": int(res.success), "termination": res.termination, "steps": res.steps,
                    "local_repairs": res.local_repairs, "full_replans": res.full_replans,
                    "stale_detections": res.stale_detections, "collisions": res.collisions,
                    "guard_violations": res.guard_violations,
                    "goal_xy_err_mm": round(res.goal_xy_err_mm, 3),
                    "decision_latency_ms": round(res.decision_latency_ms, 4),
                })
                ev_file.write(json.dumps({"task": task, "episode": ep, "seed": seed,
                                          "success": res.success, "events": res.events}) + "\n")

    write_csv(rows, run_dir / "episodes.csv")
    summary = summarise(rows)
    write_summary(summary, run_dir / "summary.json", {"config": args.config, "tag": tag})
    if not args.quiet:
        print(f"\nRun folder: {run_dir}\n")
        print_table(summary)


if __name__ == "__main__":
    main()
