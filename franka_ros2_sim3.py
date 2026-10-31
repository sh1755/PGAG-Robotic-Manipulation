
"""
PGAG Sim3: Robotic Manipulation and Failure Recovery

Franka Panda | ROS 2 Humble | PyBullet

Tasks:
    pick_place
    stack
    stack_disturb

Requires:
    franka_ros2_sim2.py
    grammar/
    
Grasping uses a simulated fixed constraint.
The disturbance is injected in simulation only.
"""

import argparse
import json

import pybullet as p
import rclpy

from std_msgs.msg import String

from franka_ros2_sim2 import FrankaStack, START
from grammar.predicates import P
from grammar.programs import TASKS


# ==========================================================
# 1. PGAG RECOVERY CONTROLLER
# ==========================================================

class FrankaRecovery(FrankaStack):

    def __init__(
        self,
        headless=False,
        repair_enabled=True
    ):

        super().__init__(headless=headless)

        self.repair_enabled = repair_enabled

        self.get_logger().info(
            "Sim3 ready: pick_place | stack | "
            "stack_disturb"
        )

        self.get_logger().info(
            f"Local repair enabled: {repair_enabled}"
        )

    # ======================================================
    # 2. ROS 2 COMMANDS
    # ======================================================

    def on_task(self, msg):

        task = msg.data.strip()

        if task in (
            "pick_place",
            "stack",
            "stack_disturb"
        ):

            self.jobs.append(task)

            self.get_logger().info(
                f"Queued task: {task}"
            )

        else:

            self.get_logger().warning(
                f"Unsupported task: {task}"
            )

    # ======================================================
    # 3. CHECK BLOCK STACKING STATE
    # ======================================================

    def on_b_on_a(self):

        # Read object poses from PyBullet.
        # Evaluate the persistent PGAG predicate.

        state = self.world()

        relation = P("on", "b", "a")

        return relation.holds(state)

    # ======================================================
    # 4. EXECUTE AND VERIFY ACTION
    # ======================================================

    def run_checked(self, atom, events):

        # Check preconditions
        pre = atom.failed_pre(self.world())

        if pre:

            raise RuntimeError(
                f"{atom.id}: missing preconditions: "
                + ", ".join(map(str, pre))
            )

        # Execute the original Sim2 action
        self.run_atom(atom)

        # Check resulting effects
        effects = atom.failed_effect(
            self.world()
        )

        if effects:

            raise RuntimeError(
                f"{atom.id}: failed effects: "
                + ", ".join(map(str, effects))
            )

        events.append({
            "event": "atom_pass",
            "atom": atom.id
        })

        self.get_logger().info(
            f"PASS: {atom.id}"
        )

    # ======================================================
    # 5. INJECT A CONTROLLED FAILURE
    # ======================================================

    def disturb_block_b(self, events):

        # The block must already be released.
        if (
            self.held is not None
            or self.constraint is not None
        ):

            raise RuntimeError(
                "Cannot disturb a held block"
            )

        body = self.objects["b"]

        before = list(self.xyz("b"))

        # Simulate a displaced block.
        # Return Block 2 to its initial location.
        p.resetBasePositionAndOrientation(
            body,
            START["b"],
            [0, 0, 0, 1]
        )

        p.resetBaseVelocity(
            body,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )

        self.step(120)

        after = list(self.xyz("b"))

        events.append({
            "event": "injected_displacement",
            "object": "b",
            "from_xyz": [
                round(float(x), 4)
                for x in before
            ],
            "to_xyz": [
                round(float(x), 4)
                for x in after
            ]
        })

        self.get_logger().warning(
            "FAULT INJECTED: "
            "Block 2 has been displaced"
        )

    # ======================================================
    # 6. LOCAL RECOVERY OF BLOCK 2
    # ======================================================

    def repair_b_placement(
        self,
        program,
        events
    ):

        # Do not attempt recovery while holding
        # another block.
        if (
            self.held is not None
            or self.constraint is not None
        ):

            raise RuntimeError(
                "Unsafe to recover Block 2 "
                "while another block is held"
            )

        # Extract only the B-on-A action chain.
        b_atoms = [
            atom
            for atom in program.atoms
            if atom.id.startswith("b_")
        ]

        if len(b_atoms) != 6:

            raise RuntimeError(
                "Unexpected stack grammar"
            )

        self.get_logger().warning(
            "LOCAL REPAIR STARTED"
        )

        self.get_logger().info(
            "Re-executing Block 2 placement only"
        )

        events.append({
            "event": "local_repair_start",
            "atoms": [
                atom.id for atom in b_atoms
            ]
        })

        for atom in b_atoms:

            self.run_checked(
                atom,
                events
            )

        # Verify the placement
        if not self.on_b_on_a():

            raise RuntimeError(
                "Recovery failed: on(b,a) false"
            )

        events.append({
            "event": "local_repair_success"
        })

        self.get_logger().info(
            "LOCAL REPAIR COMPLETE"
        )

    # ======================================================
    # 7. EXECUTE PGAG PROGRAM
    # ======================================================

    def execute(self, task):

        base_task = (
            "stack"
            if task == "stack_disturb"
            else task
        )

        program = TASKS[base_task]()

        problems = program.validate()

        if problems:

            raise RuntimeError(
                "Invalid grammar: "
                + "; ".join(problems)
            )

        self.reset_scene(base_task)

        self.run_stats = {
            "local_repairs": 0,
            "stale_detections": 0,
            "disturbance_injected": False,
            "events": []
        }

        events = self.run_stats["events"]

        for atom in program.atoms:

            # Before handling Block 3, verify
            # that Block 2 is still on Block 1.
            if atom.id.startswith("c_"):

                if not self.on_b_on_a():

                    self.run_stats[
                        "stale_detections"
                    ] += 1

                    events.append({
                        "event": "stale_effect",
                        "predicate": "on(b,a)",
                        "before_atom": atom.id
                    })

                    self.get_logger().warning(
                        "FAILURE DETECTED: "
                        "Block 2 not on Block 1"
                    )

                    if not self.repair_enabled:

                        raise RuntimeError(
                            "Placement failed; "
                            "local repair disabled"
                        )

                    self.repair_b_placement(
                        program,
                        events
                    )

                    self.run_stats[
                        "local_repairs"
                    ] += 1

            # Execute the next PGAG action
            self.run_checked(
                atom,
                events
            )

            # After the first placement,
            # inject the disturbance.
            if (
                task == "stack_disturb"
                and atom.id == "b_release"
            ):

                self.disturb_block_b(
                    events
                )

                self.run_stats[
                    "disturbance_injected"
                ] = True

        # Final goal verification
        state = self.world()

        failed_goals = [
            str(goal)
            for goal in program.goal
            if not goal.holds(state)
        ]

        if failed_goals:

            raise RuntimeError(
                "Unmet PGAG goals: "
                + ", ".join(failed_goals)
            )

        return {
            "task": task,
            "success": True,
            "local_repairs":
                self.run_stats["local_repairs"],
            "stale_detections":
                self.run_stats["stale_detections"],
            "disturbance_injected":
                self.run_stats["disturbance_injected"],
            "object_positions": {
                name: [
                    round(float(v), 4)
                    for v in obj.pos
                ]
                for name, obj
                in state.objects.items()
                if name != "bin"
            },
            "events": events,
            "grasp_model": "fixed_constraint_demo"
        }

    # ======================================================
    # 8. ROS 2 RESULT PUBLISHER
    # ======================================================

    def execute_and_publish(self, task):

        try:

            result = self.execute(task)

        except Exception as exc:

            self.get_logger().error(
                str(exc)
            )

            result = {
                "task": task,
                "success": False,
                "error": str(exc),
                "repair_enabled":
                    self.repair_enabled,
                "grasp_model":
                    "fixed_constraint_demo",
                **getattr(self, "run_stats", {})
            }

        finally:

            if self.constraint is not None:

                p.removeConstraint(
                    self.constraint
                )

                self.constraint = None
                self.held = None

                self.gripper(0.04)

        self.result_pub.publish(
            String(data=json.dumps(result))
        )

        self.get_logger().info(
            json.dumps(result)
        )

        return result


# ==========================================================
# 9. MAIN FUNCTION
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--headless",
        action="store_true"
    )

    parser.add_argument(
        "--once",
        choices=[
            "pick_place",
            "stack",
            "stack_disturb"
        ]
    )

    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable local recovery for baseline"
    )

    args = parser.parse_args()

    rclpy.init()

    node = FrankaRecovery(
        headless=args.headless,
        repair_enabled=not args.no_repair
    )

    if args.once:

        try:

            result = node.execute_and_publish(
                args.once
            )

            if not result["success"]:
                raise SystemExit(1)

        finally:

            if p.isConnected():
                p.disconnect()

            node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()

    else:

        try:
            node.spin_simulation()

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
