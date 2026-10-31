"""Run Unitree G1 29-DoF real MuJoCo physics smoke tests.

Commands from the PGAG repository root:
    python -m humanoid.g1_mujoco_probe --once arm_demo --gui
    python -m humanoid.g1_mujoco_probe --once disturb --gui
    python -m humanoid.g1_mujoco_probe --once scene --headless

This is a fixed-base robot with real MuJoCo dynamics, NOT full humanoid
balance or grasping. No real hardware connection is made.
"""
import argparse
import json

from .g1_mujoco_adapter import G1MujocoAdapter, default_model_path


def run_probe(task, *, model_path=None, gui=False, realtime=False):
    with G1MujocoAdapter(model_path, gui=gui) as sim:
        if task == "arm_demo":
            return sim.arm_demo(realtime=realtime)
        if task == "disturb":
            return sim.disturb_block_b(realtime=realtime)
        if task == "scene":
            sim.step(150, realtime=realtime)
            return {
                "task": "g1_scene",
                "success": True,
                "model": str(sim.model_path),
                "nq": sim.model.nq,
                "nu": sim.model.nu,
                "joints": sim.model.njnt,
                "wrist_xyz": [round(float(x), 4) for x in sim.wrist_xyz()],
                "on_b_a": sim.on_b_on_a(),
                "notes": "Real MuJoCo physics, fixed G1 base, no hand controller",
            }
        raise ValueError(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", choices=("scene", "arm_demo", "disturb"), default="arm_demo")
    parser.add_argument("--model-path", default=str(default_model_path()))
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--gui", action="store_true", help="Open MuJoCo GUI")
    vis.add_argument("--headless", action="store_true", help="Do not open GUI (default)")
    parser.add_argument("--realtime", action="store_true", help="Step near wall-clock speed")
    args = parser.parse_args()
    result = run_probe(
        args.once,
        model_path=args.model_path,
        gui=args.gui and not args.headless,
        realtime=args.realtime,
    )
    print(json.dumps(result, indent=2))
    if result.get("success") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
