"""Print the decision trace of one episode from a run folder.

    python -m eval.replay runs/<run>/ --task stack --episode 7
    python -m eval.replay runs/<run>/ --failures        # list all failed episodes
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--task")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args()

    for line in open(Path(args.run_dir) / "events.jsonl"):
        rec = json.loads(line)
        if args.failures:
            if not rec["success"]:
                print(f"{rec['task']:<12} episode {rec['episode']:>4}  seed {rec['seed']}")
            continue
        if rec["task"] != args.task or rec["episode"] != args.episode:
            continue
        print(f"{rec['task']} episode {rec['episode']} (seed {rec['seed']}) success={rec['success']}")
        for e in rec["events"]:
            extra = {k: v for k, v in e.items() if k not in ("step", "event", "atom") and v not in ("", None)}
            print(f"  step {e['step']:>3}  {e['event']:<22} {e['atom'] or '':<14} {extra if extra else ''}")
        return


if __name__ == "__main__":
    main()
