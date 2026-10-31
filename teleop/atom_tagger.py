"""Tag skill (atom) boundaries while you teleoperate.

Two ways to use it:

1. Standalone, in a second terminal next to your recording script:

       python teleop/atom_tagger.py --task stack --out data/tags/stack_session1.jsonl

   Press `n` at the moment you start recording an episode, then a number key
   each time a new skill starts. `s`/`f` end the episode as success/failure.

2. Inside your own recording loop (exact timestamps, no second terminal):

       from teleop.atom_tagger import AtomTagger
       tagger = AtomTagger("stack", "data/tags/stack.jsonl")
       tagger.start_episode(); tagger.tag("APPROACH"); ...; tagger.end_episode("success")

Keys:  1 APPROACH  2 GRASP  3 LIFT  4 TRANSPORT  5 ALIGN  6 INSERT  7 RELEASE  8 RECOVER
       n new episode   s end: success   f end: failure   u undo last tag   q quit

Output: one JSON line per episode with wall-clock segment times. Convert to
per-frame labels with teleop/align_tags.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

KEYMAP = {"1": "APPROACH", "2": "GRASP", "3": "LIFT", "4": "TRANSPORT",
          "5": "ALIGN", "6": "INSERT", "7": "RELEASE", "8": "RECOVER"}


class AtomTagger:
    def __init__(self, task: str, out: str):
        self.task, self.out = task, Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.episode = self._next_episode_index()
        self.t0: Optional[float] = None
        self.tags: List[dict] = []

    def _next_episode_index(self) -> int:
        if not self.out.exists():
            return 0
        return sum(1 for _ in open(self.out))

    @property
    def active(self) -> bool:
        return self.t0 is not None

    def start_episode(self) -> None:
        self.t0, self.tags = time.time(), []

    def tag(self, skill: str) -> None:
        if not self.active:
            raise RuntimeError("start_episode() first")
        self.tags.append({"skill": skill, "t": time.time()})

    def undo(self) -> Optional[str]:
        return self.tags.pop()["skill"] if self.tags else None

    def end_episode(self, outcome: str) -> dict:
        t_end = time.time()
        segs = []
        for i, tg in enumerate(self.tags):
            t_next = self.tags[i + 1]["t"] if i + 1 < len(self.tags) else t_end
            segs.append({"skill": tg["skill"], "t_start": tg["t"] - self.t0, "t_end": t_next - self.t0})
        rec = {"task": self.task, "episode": self.episode, "outcome": outcome,
               "t_start_wall": self.t0, "duration_s": t_end - self.t0, "segments": segs}
        with open(self.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.episode += 1
        self.t0, self.tags = None, []
        return rec


def _getch() -> str:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tg = AtomTagger(args.task, args.out)
    print(__doc__.split("Keys:")[1].split("Output:")[0])
    print(f"Next episode index: {tg.episode}. Press n to start.")
    while True:
        k = _getch()
        if k in ("q", "\x03"):
            if tg.active:
                print("\r\nEpisode in progress was discarded.")
            break
        if k == "n":
            tg.start_episode()
            print(f"\r\n[episode {tg.episode}] started")
        elif not tg.active:
            print("\r\n(press n to start an episode)")
        elif k in KEYMAP:
            tg.tag(KEYMAP[k])
            print(f"\r  {time.time() - tg.t0:6.2f}s  {KEYMAP[k]}")
        elif k == "u":
            print(f"\r  undone: {tg.undo()}")
        elif k in ("s", "f"):
            rec = tg.end_episode("success" if k == "s" else "failure")
            seq = " > ".join(s["skill"] for s in rec["segments"])
            print(f"\r[episode {rec['episode']}] {rec['outcome']}, {rec['duration_s']:.1f}s: {seq}")
    print("\r\nSaved to", args.out)


if __name__ == "__main__":
    main()
