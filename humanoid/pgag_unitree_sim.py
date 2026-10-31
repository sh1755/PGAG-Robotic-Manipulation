"""PGAG humanoid entry point (working MOCK mode, real G1 adapter TODO).

From the existing PGAG root:
    python -m humanoid.pgag_unitree_sim --backend mock --task stack
    python -m humanoid.pgag_unitree_sim --backend mock --task stack --disturb
    python -m humanoid.pgag_unitree_sim --backend mock --task stack --disturb --no-repair

All mock-mode results are symbolic, NOT MuJoCo physics experiments.
"""

import argparse
import json

from grammar.programs import TASKS

from .g1_failure_recovery import G1LocalRecovery
from .g1_state_monitor import G1StateMonitor
from .mock_g1_adapter import MockG1Adapter
from .unitree_g1_adapter import UnitreeG1Adapter


def execute_pgag(task="stack", disturb=False, repair=True, backend="mock"):
    if backend not in ("mock", "unitree-mujoco"):
        raise ValueError(f"Unknown backend: {backend}")
    if disturb and task != "stack":
        raise ValueError("The demonstration disturbance is defined for stack only.")

    adapter = MockG1Adapter() if backend == "mock" else UnitreeG1Adapter()
    program = TASKS[task]()
    issues = program.validate()
    if issues:
        raise RuntimeError(f"Invalid task grammar: {issues}")

    result = {
        "task": task,
        "backend": backend,
        "simulation_type": "idealised_symbolic_mock" if backend == "mock" else "not_implemented",
        "success": False,
        "repair_enabled": repair,
        "disturbance": bool(disturb),
        "stale_detections": 0,
        "local_repairs": 0,
        "executed_atoms": [],
        "events": [],
    }

    # Real Unitree/MuJoCo integration fails loudly rather than pretending to run.
    adapter.reset(task)
    monitor = G1StateMonitor(adapter)

    def run_checked(atom):
        pre = monitor.unmet(atom.pre)
        if pre:
            raise RuntimeError(f"{atom.id}: unmet preconditions: {pre}")
        adapter.execute_atom(atom)
        effects = monitor.unmet(atom.effect)
        if effects:
            raise RuntimeError(f"{atom.id}: unmet effects: {effects}")
        result["executed_atoms"].append(atom.id)

    repair_engine = G1LocalRecovery()
    try:
        for atom in program.atoms:
            # Stale-state detection before running the next dependent task.
            if task == "stack" and atom.id.startswith("c_"):
                if not monitor.on("b", "a"):
                    result["stale_detections"] += 1
                    result["events"].append("stale_effect:on(b,a)")
                    if not repair:
                        raise RuntimeError("on(b,a) false and repair disabled")
                    chain = repair_engine.repair_on_b_a(program, run_checked)
                    result["local_repairs"] += 1
                    result["events"].append({"replayed": chain})
                    if not monitor.on("b", "a"):
                        raise RuntimeError("Repair failed to restore on(b,a)")

            run_checked(atom)
            if disturb and atom.id == "b_release":
                adapter.inject_displacement("b")
                result["events"].append("injected_displacement:b")

        result["missing_goals"] = monitor.unmet(program.goal)
        result["success"] = len(result["missing_goals"]) == 0

    except Exception as exc:
        result["error"] = str(exc)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("stack", "pick_place"), default="stack")
    parser.add_argument("--backend", choices=("mock", "unitree-mujoco"), default="mock")
    parser.add_argument("--disturb", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()

    try:
        result = execute_pgag(
            task=args.task,
            backend=args.backend,
            disturb=args.disturb,
            repair=not args.no_repair,
        )
    except NotImplementedError as exc:
        parser.exit(2, f"Adapter incomplete: {exc}\n")

    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
