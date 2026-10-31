"""Convert atom tags (from atom_tagger.py) into per-frame skill labels.

    python teleop/align_tags.py data/tags/stack.jsonl --fps 30 --out data/tags/stack_frames.json
    python teleop/align_tags.py data/tags/stack.jsonl --fps 30 --offset 0.15   # tagger pressed 0.15 s late
    python teleop/align_tags.py data/tags/stack.jsonl --fps 30 --check stack   # compare with grammar program

Each frame i of an episode has time i / fps from the episode start. Frames
before the first tag are labelled "IDLE". Use --n-frames-file (JSON list of
frame counts per episode, e.g. from your LeRobot dataset) to cut labels to
the exact recorded length.

--check compares each tagged skill sequence against the skills of the grammar
program for that task, and reports episodes whose sequence deviates (useful for
spotting mistagged demos, or genuine recoveries you want to keep).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def labels_for(rec: dict, fps: float, offset: float, n_frames: int | None) -> list:
    n = n_frames if n_frames is not None else int(math.ceil(rec["duration_s"] * fps))
    out = ["IDLE"] * n
    for seg in rec["segments"]:
        a = max(0, int(round((seg["t_start"] - offset) * fps)))
        b = min(n, int(round((seg["t_end"] - offset) * fps)))
        for i in range(a, b):
            out[i] = seg["skill"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--offset", type=float, default=0.0, help="seconds to subtract from tag times")
    ap.add_argument("--n-frames-file", help="JSON list of frame counts, one per episode")
    ap.add_argument("--out")
    ap.add_argument("--check", metavar="TASK", help="compare skill sequences with this grammar program")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.tags) if l.strip()]
    counts = json.load(open(args.n_frames_file)) if args.n_frames_file else [None] * len(recs)

    if args.check:
        from grammar.programs import TASKS
        expected = [a.skill.value for a in TASKS[args.check]().atoms]
        print(f"Expected: {' > '.join(expected)}")
        for r in recs:
            got = [s["skill"] for s in r["segments"]]
            flag = "ok " if got == expected else "DIFF"
            print(f"  [{flag}] ep {r['episode']:>3} {r['outcome']:<8} {' > '.join(got)}")

    if args.out:
        result = [{"episode": r["episode"], "task": r["task"], "outcome": r["outcome"], "fps": args.fps,
                   "labels": labels_for(r, args.fps, args.offset, c)} for r, c in zip(recs, counts)]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"))
        print(f"Wrote per-frame labels for {len(result)} episodes to {args.out}")


if __name__ == "__main__":
    main()
