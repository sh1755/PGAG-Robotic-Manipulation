
"""
PGAG: Physics-Grounded Action Grammar
Franka Panda | ROS 2 Humble | PyBullet

Tasks:
  pick_place: Single cube manipulation
  stack: Block 2 on Block 1,
         then Block 3 on Block 2

Run from the existing pgag directory.

Grasping uses a simulated fixed constraint.
Automatic local repair is not implemented here.
"""

import argparse
import json
import math
import time
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data
import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from grammar.atoms import Skill
from grammar.predicates import ObjectState, WorldState
from grammar.programs import TASKS


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DT = 1.0 / 240.0
SIDE = 0.05

HOME = [0, -0.4, 0, -2, 0, 1.8, 0.8]

OPEN = 0.04
CLOSED = 0.025

START = {
    "cube": [0.55, 0.0, SIDE / 2],
    "a": [0.46, 0.18, SIDE / 2],
    "b": [0.52, -0.16, SIDE / 2],
    "c": [0.61, -0.02, SIDE / 2],
}

DESTINATION = [0.40, 0.25, 0.05]

LABELS = {
    "a": "Block 1",
    "b": "Block 2",
    "c": "Block 3",
    "cube": "Cube"
}


# ==========================================================
# 2. ROS 2 + PYBULLET NODE
# ==========================================================

class FrankaStack(Node):

    def __init__(self, headless=False):

        super().__init__("pgag_franka_stack")

        self.headless = headless
        self.jobs = deque()

        self.create_subscription(
            String,
            "/pgag/task",
            self.on_task,
            10
        )

        self.result_pub = self.create_publisher(
            String,
            "/pgag/physics_result",
            10
        )

        mode = p.DIRECT if headless else p.GUI

        if p.connect(mode) < 0:
            raise RuntimeError(
                "Failed to open PyBullet"
            )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)

        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True
        )

        self.arm = list(range(7))
        self.fingers = [9, 10]

        # Panda grasp target link
        self.ee = 11

        self.down = p.getQuaternionFromEuler(
            [math.pi, 0, 0]
        )

        self.objects = {}

        self.held = None
        self.constraint = None

        self.reset_robot()

        if not headless:
            p.resetDebugVisualizerCamera(
                1.4,
                50,
                -35,
                [0.45, 0.0, 0.19]
            )

        self.get_logger().info(
            "Ready: publish stack or pick_place "
            "to /pgag/task"
        )

    # ======================================================
    # 3. ROS 2 CALLBACK
    # ======================================================

    def on_task(self, msg):

        name = msg.data.strip()

        if name in ("stack", "pick_place"):

            self.jobs.append(name)

            self.get_logger().info(
                f"Received task: {name}"
            )

        else:

            self.get_logger().warning(
                f"Unsupported task: {name}"
            )

    # ======================================================
    # 4. PHYSICS SIMULATION STEP
    # ======================================================

    def step(self, n=120):

        for _ in range(n):

            p.stepSimulation()

            if not self.headless:
                time.sleep(DT)

    # ======================================================
    # 5. RESET FRANKA PANDA
    # ======================================================

    def reset_robot(self):

        for j, q in enumerate(HOME):

            p.resetJointState(
                self.robot,
                j,
                q,
                targetVelocity=0
            )

            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=q,
                force=87
            )

        for j in self.fingers:

            p.resetJointState(
                self.robot,
                j,
                OPEN
            )

            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=OPEN,
                force=40
            )

    # ======================================================
    # 6. CREATE COLOURED BLOCK
    # ======================================================

    def make_box(
        self,
        name,
        xyz,
        colour,
        mass=0.1
    ):

        half = [SIDE / 2] * 3

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=colour
        )

        body = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=xyz
        )

        p.changeDynamics(
            body,
            -1,
            lateralFriction=1.2,
            restitution=0
        )

        self.objects[name] = body

        # Show Block 1 / Block 2 / Block 3
        if not self.headless:

            p.addUserDebugText(
                LABELS.get(name, name),
                [0, 0, SIDE + 0.025],
                [0, 0, 0],
                textSize=1.2,
                parentObjectUniqueId=body,
                parentLinkIndex=-1
            )

    # ======================================================
    # 7. RESET SCENE FOR EACH TASK
    # ======================================================

    def reset_scene(self, task):

        if self.constraint is not None:
            p.removeConstraint(self.constraint)

        self.constraint = None
        self.held = None

        if not self.headless:
            p.removeAllUserDebugItems()

        for body in self.objects.values():
            p.removeBody(body)

        self.objects.clear()

        self.reset_robot()

        if task == "stack":

            # Base block: red
            self.make_box(
                "a",
                START["a"],
                [0.90, 0.15, 0.12, 1],
                mass=0
            )

            # Block 2: blue
            self.make_box(
                "b",
                START["b"],
                [0.10, 0.37, 0.94, 1]
            )

            # Block 3: yellow
            self.make_box(
                "c",
                START["c"],
                [0.95, 0.72, 0.12, 1]
            )

        else:

            self.make_box(
                "cube",
                START["cube"],
                [0.9, 0.13, 0.13, 1]
            )

            # Visual target marker
            marker = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    0.07, 0.07, 0.001
                ],
                rgbaColor=[
                    0.1, 0.8, 0.2, 0.5
                ]
            )

            self.objects["_target_marker"] = (
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=marker,
                    basePosition=[
                        DESTINATION[0],
                        DESTINATION[1],
                        0.001
                    ]
                )
            )

        self.step(120)

        self.get_logger().info(
            f"Scene reset: {task}"
        )

    # ======================================================
    # 8. OBJECT AND END-EFFECTOR POSES
    # ======================================================

    def pose(self, name):

        return p.getBasePositionAndOrientation(
            self.objects[name]
        )

    def xyz(self, name):

        return self.pose(name)[0]

    def tcp(self):

        return p.getLinkState(
            self.robot,
            self.ee,
            computeForwardKinematics=True
        )[4]

    # ======================================================
    # 9. PHYSICS-GROUNDED WORLD STATE
    # ======================================================

    def world(self):

        objects = {}

        for name, body in self.objects.items():

            if name.startswith("_"):
                continue

            xyz, q = (
                p.getBasePositionAndOrientation(
                    body
                )
            )

            objects[name] = ObjectState(
                pos=np.array(xyz),
                size=np.array([SIDE] * 3),
                quat=np.array([
                    q[3],
                    q[0],
                    q[1],
                    q[2]
                ])
            )

        if "cube" in objects:

            objects["bin"] = ObjectState(
                pos=np.array(DESTINATION),
                size=np.array([
                    0.14, 0.14, 0.10
                ]),
                kind="container"
            )

        return WorldState(
            objects=objects,
            ee_pos=np.array(self.tcp()),
            held=self.held,
            gripper_width=2 * (
                CLOSED if self.held else OPEN
            )
        )

    # ======================================================
    # 10. INVERSE KINEMATICS
    # ======================================================

    def move(self, target, tol=0.018):

        target = [
            float(v) for v in target
        ]

        for attempt in range(4):

            q = p.calculateInverseKinematics(
                self.robot,
                self.ee,
                target,
                targetOrientation=self.down,
                maxNumIterations=250,
                residualThreshold=1e-5
            )

            for _ in range(360):

                p.setJointMotorControlArray(
                    self.robot,
                    self.arm,
                    p.POSITION_CONTROL,
                    targetPositions=q[:7],
                    forces=[
                        87, 87, 87, 87,
                        12, 12, 12
                    ]
                )

                self.step(1)

            error = math.dist(
                target,
                self.tcp()
            )

            if error <= tol:
                return error

        raise RuntimeError(
            f"IK failed: target={target}, "
            f"error={error:.3f} m"
        )

    # ======================================================
    # 11. GRIPPER
    # ======================================================

    def gripper(self, width):

        for j in self.fingers:

            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=width,
                force=40
            )

        self.step(120)

    # ======================================================
    # 12. MOVE A HELD OBJECT
    # ======================================================

    def move_held(
        self,
        name,
        desired_center,
        tol=0.018
    ):

        if self.held != name:

            raise RuntimeError(
                f"Not holding {name}"
            )

        for _ in range(4):

            center = np.asarray(
                self.xyz(name)
            )

            actual_tcp = np.asarray(
                self.tcp()
            )

            offset = actual_tcp - center

            goal_tcp = (
                np.asarray(desired_center)
                + offset
            )

            self.move(
                goal_tcp,
                tol=tol
            )

            difference = np.linalg.norm(
                np.asarray(self.xyz(name))
                - desired_center
            )

            if difference <= tol:
                return float(difference)

        raise RuntimeError(
            f"Object {name} offset "
            f"from goal: {difference:.3f} m"
        )

    # ======================================================
    # 13. SIMULATED GRASP ATTACHMENT
    # ======================================================

    def attach(self, name):

        link = p.getLinkState(
            self.robot,
            self.ee,
            computeForwardKinematics=True
        )

        tcp_xyz = link[4]
        tcp_q = link[5]

        obj_xyz, obj_q = self.pose(name)

        dist = math.dist(
            tcp_xyz,
            obj_xyz
        )

        self.get_logger().info(
            f"Grasp alignment {name}: "
            f"{dist * 1000:.1f} mm"
        )

        if dist > 0.035:

            raise RuntimeError(
                f"Cannot grasp {name}: "
                f"TCP-to-cube {dist:.3f} m"
            )

        inv_xyz, inv_q = p.invertTransform(
            tcp_xyz,
            tcp_q
        )

        local_xyz, local_q = (
            p.multiplyTransforms(
                inv_xyz,
                inv_q,
                obj_xyz,
                obj_q
            )
        )

        self.constraint = p.createConstraint(
            self.robot,
            self.ee,
            self.objects[name],
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            local_xyz,
            [0, 0, 0],
            parentFrameOrientation=local_q
        )

        if self.constraint < 0:

            raise RuntimeError(
                "Could not create grasp constraint"
            )

        self.held = name

        self.get_logger().info(
            f"Simulated attachment: {name}"
        )

    # ======================================================
    # 14. EXECUTE PGAG ACTION
    # ======================================================

    def run_atom(self, atom):

        name = atom.obj
        target = atom.target
        skill = atom.skill

        self.get_logger().info(
            f"{atom.id}: "
            f"{skill.value} {atom.objects}"
        )

        # APPROACH
        if skill == Skill.APPROACH:

            self.gripper(OPEN)

            x, y, z = self.xyz(name)

            self.move([
                x,
                y,
                max(z + 0.23, 0.28)
            ])

            x, y, z = self.xyz(name)

            self.move(
                [x, y, z + 0.003],
                tol=0.012
            )

        # GRASP
        elif skill == Skill.GRASP:

            self.gripper(CLOSED)
            self.attach(name)

        # LIFT
        elif skill == Skill.LIFT:

            x, y, z = self.xyz(name)

            self.move_held(
                name,
                [
                    x,
                    y,
                    max(z + 0.22, 0.28)
                ]
            )

        # TRANSPORT
        elif skill == Skill.TRANSPORT:

            destination = (
                DESTINATION
                if target == "bin"
                else self.xyz(target)
            )

            self.move_held(
                name,
                [
                    destination[0],
                    destination[1],
                    0.31
                ],
                tol=0.012
            )

        # ALIGN
        elif skill == Skill.ALIGN:

            target_pos = self.xyz(target)

            top = (
                target_pos[2] + SIDE / 2
            )

            self.move_held(
                name,
                [
                    target_pos[0],
                    target_pos[1],
                    top + SIDE / 2 + 0.055
                ],
                tol=0.004
            )

        # RELEASE
        elif skill == Skill.RELEASE:

            if (
                self.held != name
                or self.constraint is None
            ):

                raise RuntimeError(
                    f"Release failed: "
                    f"not holding {name}"
                )

            target_pos = (
                DESTINATION
                if target == "bin"
                else self.xyz(target)
            )

            surface = (
                0.0
                if target == "bin"
                else target_pos[2] + SIDE / 2
            )

            self.move_held(
                name,
                [
                    target_pos[0],
                    target_pos[1],
                    surface + SIDE / 2 + 0.008
                ],
                tol=0.012
            )

            p.removeConstraint(
                self.constraint
            )

            self.constraint = None
            self.held = None

            self.gripper(OPEN)

            self.move(
                [
                    target_pos[0],
                    target_pos[1],
                    0.34
                ],
                tol=0.025
            )

            self.step(240)

        else:

            raise NotImplementedError(
                f"Physics action {skill.value} "
                f"not implemented"
            )

    # ======================================================
    # 15. EXECUTE GRAMMAR PROGRAM
    # ======================================================

    def execute(self, task):

        program = TASKS[task]()

        issues = program.validate()

        if issues:

            raise RuntimeError(
                "Invalid PGAG grammar: "
                + "; ".join(issues)
            )

        self.reset_scene(task)

        for atom in program.atoms:

            failed_pre = atom.failed_pre(
                self.world()
            )

            if failed_pre:

                raise RuntimeError(
                    f"{atom.id} missing "
                    f"preconditions: {failed_pre}"
                )

            self.run_atom(atom)

            failed_effects = atom.failed_effect(
                self.world()
            )

            if failed_effects:

                raise RuntimeError(
                    f"{atom.id} missing "
                    f"effects: {failed_effects}"
                )

            self.get_logger().info(
                f"PASS: {atom.id}"
            )

        state = self.world()

        failed_goals = [
            str(goal)
            for goal in program.goal
            if not goal.holds(state)
        ]

        return {
            "task": task,
            "success": not failed_goals,
            "failed_goals": failed_goals,
            "object_positions": {
                n: [
                    round(float(v), 4)
                    for v in ob.pos
                ]
                for n, ob in state.objects.items()
                if n != "bin"
            },
            "grasp_model": "fixed_constraint_demo"
        }

    # ======================================================
    # 16. PUBLISH EXECUTION RESULTS
    # ======================================================

    def execute_and_publish(self, task):

        try:

            outcome = self.execute(task)

        except Exception as e:

            self.get_logger().error(
                str(e)
            )

            outcome = {
                "task": task,
                "success": False,
                "error": str(e),
                "grasp_model": "fixed_constraint_demo"
            }

        finally:

            if self.constraint is not None:

                p.removeConstraint(
                    self.constraint
                )

                self.constraint = None
                self.held = None

                self.gripper(OPEN)

        self.result_pub.publish(
            String(data=json.dumps(outcome))
        )

        self.get_logger().info(
            json.dumps(outcome)
        )

        return outcome

    # ======================================================
    # 17. SIMULATION MAIN LOOP
    # ======================================================

    def spin_simulation(self):

        try:

            while rclpy.ok() and p.isConnected():

                rclpy.spin_once(
                    self,
                    timeout_sec=0
                )

                if self.jobs:

                    self.execute_and_publish(
                        self.jobs.popleft()
                    )

                else:

                    self.step(1)

        finally:

            if p.isConnected():
                p.disconnect()

            self.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


# ==========================================================
# 18. MAIN FUNCTION
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without PyBullet GUI"
    )

    parser.add_argument(
        "--once",
        choices=["stack", "pick_place"],
        help="Run one task without a ROS 2 command"
    )

    args = parser.parse_args()

    rclpy.init()

    node = FrankaStack(
        headless=args.headless
    )

    if args.once:

        try:

            node.execute_and_publish(
                args.once
            )

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
