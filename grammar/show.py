"""Print a task program, validate it, and optionally export its dependency graph.

    python -m grammar.show --task stack
    python -m grammar.show --task stack --dot stack.dot   # then: dot -Tpng stack.dot -o stack.png
"""
import argparse
import sys

from .programs import TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default=None, help="default: all tasks")
    ap.add_argument("--dot", default=None, help="write Graphviz DOT file (single task only)")
    args = ap.parse_args()

    names = [args.task] if args.task else list(TASKS)
    ok = True
    for name in names:
        prog = TASKS[name]()
        print(prog.summary())
        problems = prog.validate()
        print("VALID" if not problems else "INVALID:\n  " + "\n  ".join(problems))
        print()
        ok &= not problems
        if args.dot and args.task:
            with open(args.dot, "w") as f:
                f.write(prog.to_dot())
            print(f"Wrote {args.dot}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
