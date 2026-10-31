
"""
PGAG: Unitree G1 MuJoCo Physics Adapter

Uses Unitree's official 29-DoF G1 MJCF model.

Features:
- Load the G1 humanoid robot
- Fixed-base right-arm simulation
- Torque-controlled shoulder and elbow movements
- Create a tabletop and three coloured blocks
- Monitor object positions
- Detect a displaced block using PGAG predicates

This version fixes the MuJoCo free-joint indexing
error in disturb_block_b().

No physical grasping, balance control or real
Unitree hardware control is implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import math
import time

import numpy as np

from grammar.predicates import (
    ObjectState,
    WorldState,
    P,
)

try:
    import mujoco
except ImportError:
    mujoco = None


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BLOCK_SIZE = 0.05

ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


# ==========================================================
# 2. UNITREE G1 MODEL PATH
# ==========================================================

def default_model_path() -> Path:

    return (
        Path(__file__).resolve().parents[3]
        / "unitree_mujoco"
        / "unitree_robots"
        / "g1"
        / "scene_29dof.xml"
    )


# ==========================================================
# 3. UNITREE G1 MUJOCO ADAPTER
# ==========================================================

class G1MujocoAdapter:

    def __init__(
        self,
        model_path: Optional[str] = None,
        gui: bool = False,
    ):

        if mujoco is None:

            raise RuntimeError(
                "MuJoCo is not installed. "
                "Run: python -m pip install "
                "'mujoco>=3.3,<4'"
            )

        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path
            else default_model_path()
        )

        if not self.model_path.is_file():

            raise FileNotFoundError(
                f"G1 model not found: {self.model_path}\n"
                "Clone the official unitree_mujoco "
                "repository beside pgag_week1."
            )

        if not hasattr(mujoco, "MjSpec"):

            raise RuntimeError(
                "Upgrade MuJoCo to a version "
                "supporting MjSpec."
            )

        # ==================================================
        # LOAD OFFICIAL UNITREE G1 MODEL
        # ==================================================

        spec = mujoco.MjSpec.from_file(
            str(self.model_path)
        )

        # Remove floating-base joint for the initial
        # stationary arm-control experiment.
        floating = spec.joint(
            "floating_base_joint"
        )

        if floating is None:

            raise RuntimeError(
                "Expected floating_base_joint "
                "in Unitree G1 model."
            )

        spec.delete(floating)

        # ==================================================
        # ADD TABLE
        # ==================================================

        spec.worldbody.add_geom(
            name="pgag_table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.52, -0.20, 0.62],
            size=[0.35, 0.32, 0.08],
            rgba=[0.48, 0.46, 0.43, 1.0],
        )

        # ==================================================
        # ADD THREE BLOCKS
        # ==================================================

        locations = {

            "a": (
                [0.44, -0.25, 0.725],
                [0.92, 0.15, 0.12, 1.0],
            ),

            "b": (
                [0.44, -0.25, 0.775],
                [0.12, 0.35, 0.95, 1.0],
            ),

            "c": (
                [0.61, -0.10, 0.725],
                [0.96, 0.73, 0.10, 1.0],
            ),
        }

        for name, (position, colour) in (
            locations.items()
        ):

            body = spec.worldbody.add_body(
                name=f"pgag_{name}",
                pos=position,
            )

            # Block 1 remains fixed.
            # Blocks 2 and 3 are movable.
            if name != "a":

                body.add_freejoint(
                    name=f"pgag_{name}_free"
                )

            body.add_geom(
                name=f"pgag_{name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[BLOCK_SIZE / 2] * 3,
                rgba=colour,
                mass=0.08 if name != "a" else 0,
                friction=[
                    1.0,
                    0.005,
                    0.0001,
                ],
            )

        # ==================================================
        # COMPILE PHYSICS MODEL
        # ==================================================

        self.model = spec.compile()

        self.model.opt.timestep = 0.002

        self.data = mujoco.MjData(
            self.model
        )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.initial_qpos = (
            self.data.qpos.copy()
        )

        self.initial_qvel = (
            self.data.qvel.copy()
        )

        # ==================================================
        # FIND ROBOT MOTORS
        # ==================================================

        self.targets: Dict[int, float] = {}

        self.motor_by_joint: Dict[
            str, int
        ] = {}

        for actuator_id in range(
            self.model.nu
        ):

            joint_id = int(
                self.model.actuator_trnid[
                    actuator_id, 0
                ]
            )

            if joint_id < 0:
                continue

            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if joint_name:

                self.motor_by_joint[
                    joint_name
                ] = actuator_id

                qpos_index = int(
                    self.model.jnt_qposadr[
                        joint_id
                    ]
                )

                self.targets[
                    joint_id
                ] = float(
                    self.data.qpos[
                        qpos_index
                    ]
                )

        missing = [
            name
            for name in ARM_JOINTS
            if name not in self.motor_by_joint
        ]

        if missing:

            raise RuntimeError(
                f"Missing G1 arm motors: {missing}"
            )

        self.model.body(
            "right_wrist_yaw_link"
        )

        # ==================================================
        # GUI CONFIGURATION
        # ==================================================

        self.gui = bool(gui)

        self.viewer = None

        self._viewer_context = None

    # ======================================================
    # 4. MUJOCO VIEWER
    # ======================================================

    def __enter__(self):

        if self.gui:

            import mujoco.viewer

            self._viewer_context = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
            )

            self.viewer = (
                self._viewer_context.__enter__()
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        if self._viewer_context is not None:

            self._viewer_context.__exit__(
                exc_type,
                exc,
                tb,
            )

            self._viewer_context = None

            self.viewer = None

    # ======================================================
    # 5. READ JOINT POSITION
    # ======================================================

    def joint_position(
        self,
        name: str,
    ) -> float:

        joint_id = self.model.joint(
            name
        ).id

        qpos_index = int(
            self.model.jnt_qposadr[
                joint_id
            ]
        )

        return float(
            self.data.qpos[qpos_index]
        )

    # ======================================================
    # 6. SET JOINT TARGET
    # ======================================================

    def set_joint_goal(
        self,
        name: str,
        radians: float,
    ) -> None:

        if name not in self.motor_by_joint:

            raise KeyError(
                f"Joint not actuated: {name}"
            )

        joint = self.model.joint(
            name
        )

        joint_id = joint.id

        if bool(
            self.model.jnt_limited[joint_id]
        ):

            low, high = map(
                float,
                self.model.jnt_range[
                    joint_id
                ],
            )

            if not low <= radians <= high:

                raise ValueError(
                    f"{name}: {radians:.3f} "
                    f"outside [{low:.3f}, "
                    f"{high:.3f}]"
                )

        self.targets[
            joint_id
        ] = float(radians)

    # ======================================================
    # 7. GET RIGHT WRIST POSITION
    # ======================================================

    def wrist_xyz(self):

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        body_id = self.model.body(
            "right_wrist_yaw_link"
        ).id

        return self.data.xpos[
            body_id
        ].copy()

    # ======================================================
    # 8. PHYSICS WORLD STATE
    # ======================================================

    def world_state(
        self,
    ) -> WorldState:

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        objects = {}

        for name in (
            "a",
            "b",
            "c",
        ):

            body_id = self.model.body(
                f"pgag_{name}"
            ).id

            objects[name] = ObjectState(
                pos=self.data.xpos[
                    body_id
                ].copy(),

                size=np.array(
                    [BLOCK_SIZE] * 3
                ),

                quat=self.data.xquat[
                    body_id
                ].copy(),
            )

        return WorldState(
            objects=objects,
            ee_pos=self.wrist_xyz(),
            held=None,
            gripper_width=0.0,
        )

    # ======================================================
    # 9. CHECK PGAG STACKING PREDICATE
    # ======================================================

    def on_b_on_a(self) -> bool:

        state = self.world_state()

        predicate = P(
            "on",
            "b",
            "a",
        )

        return bool(
            predicate.holds(state)
        )

    # ======================================================
    # 10. MUJOCO PHYSICS STEP
    # ======================================================

    def step(
        self,
        n=1,
        realtime=False,
    ):

        dt = float(
            self.model.opt.timestep
        )

        for i in range(n):

            for joint_id, target in (
                self.targets.items()
            ):

                joint_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id,
                )

                actuator_id = (
                    self.motor_by_joint[
                        joint_name
                    ]
                )

                qpos_index = int(
                    self.model.jnt_qposadr[
                        joint_id
                    ]
                )

                qvel_index = int(
                    self.model.jnt_dofadr[
                        joint_id
                    ]
                )

                current_position = float(
                    self.data.qpos[
                        qpos_index
                    ]
                )

                current_velocity = float(
                    self.data.qvel[
                        qvel_index
                    ]
                )

                is_arm = (
                    "shoulder" in joint_name
                    or "elbow" in joint_name
                    or "wrist" in joint_name
                )

                if is_arm:

                    kp = 38.0
                    kd = 2.8

                else:

                    kp = 70.0
                    kd = 4.5

                torque = (
                    kp * (
                        target - current_position
                    )
                    - kd * current_velocity
                )

                low, high = (
                    self.model.actuator_ctrlrange[
                        actuator_id
                    ]
                )

                torque = np.clip(
                    torque,
                    low,
                    high,
                )

                self.data.ctrl[
                    actuator_id
                ] = torque

            mujoco.mj_step(
                self.model,
                self.data,
            )

            if (
                self.viewer is not None
                and i % 10 == 0
            ):

                self.viewer.sync()

            if realtime:

                time.sleep(dt)

        if self.viewer is not None:

            self.viewer.sync()

    # ======================================================
    # 11. RIGHT ARM MOTION DEMO
    # ======================================================

    def arm_demo(
        self,
        realtime=False,
    ):

        self.step(
            int(
                0.4
                / self.model.opt.timestep
            ),
            realtime,
        )

        initial_xyz = list(
            map(
                float,
                self.wrist_xyz(),
            )
        )

        self.set_joint_goal(
            "right_shoulder_pitch_joint",
            0.40,
        )

        self.set_joint_goal(
            "right_elbow_joint",
            0.45,
        )

        self.step(
            int(
                2.4
                / self.model.opt.timestep
            ),
            realtime,
        )

        desired = (
            0.40,
            0.45,
        )

        observed = (
            self.joint_position(
                "right_shoulder_pitch_joint"
            ),

            self.joint_position(
                "right_elbow_joint"
            ),
        )

        error = math.sqrt(
            sum(
                (q - t) ** 2
                for q, t in zip(
                    observed,
                    desired,
                )
            )
        )

        return {
            "task": "g1_arm_demo",
            "model": self.model_path.name,
            "physics_engine": "mujoco",
            "base": "fixed_for_arm_testing",
            "joint_control": "torque_pd",
            "success": error <= 0.15,
            "joint_error_rad": round(
                error, 5
            ),
            "shoulder_pitch_rad": round(
                observed[0], 5
            ),
            "elbow_rad": round(
                observed[1], 5
            ),
            "wrist_xyz_before": [
                round(v, 4)
                for v in initial_xyz
            ],
            "wrist_xyz_after": [
                round(float(v), 4)
                for v in self.wrist_xyz()
            ],
            "caution": (
                "No physical grasp or humanoid "
                "balance validated"
            ),
        }

    # ======================================================
    # 12. BLOCK DISPLACEMENT AND FAILURE DETECTION
    # ======================================================

    def disturb_block_b(
        self,
        distance_m=0.09,
        realtime=False,
    ):

        if abs(distance_m) > 0.15:

            raise ValueError(
                "Displacement must be "
                "within 0.15 m"
            )

        # Settle physics before disturbance.
        self.step(
            int(
                0.35
                / self.model.opt.timestep
            ),
            realtime,
        )

        # Check initial PGAG relation.
        before = self.on_b_on_a()

        # Block B has a free joint.
        joint = self.model.joint(
            "pgag_b_free"
        )

        joint_id = joint.id

        # FIX: Convert MuJoCo joint addresses
        # to scalar integers before indexing.
        qpos_index = int(
            self.model.jnt_qposadr[
                joint_id
            ]
        )

        qvel_index = int(
            self.model.jnt_dofadr[
                joint_id
            ]
        )

        # Free joint position is stored as:
        # [x, y, z, qw, qx, qy, qz]
        #
        # Displace Block B along world X.
        self.data.qpos[
            qpos_index
        ] += distance_m

        # Free joint velocity has six entries:
        # [vx, vy, vz, wx, wy, wz]
        self.data.qvel[
            qvel_index:
            qvel_index + 6
        ] = 0.0

        # Recompute the scene transforms.
        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # Allow physical settling.
        self.step(
            int(
                0.20
                / self.model.opt.timestep
            ),
            realtime,
        )

        # Check if the stacking predicate
        # has become false.
        after = self.on_b_on_a()

        return {
            "task": (
                "physical_block_displacement"
            ),
            "physics_engine": "mujoco",
            "on_b_a_before": bool(
                before
            ),
            "on_b_a_after": bool(
                after
            ),
            "displacement_m": (
                distance_m
            ),
            "stale_detected": bool(
                before and not after
            ),
            "note": (
                "Block displaced through a "
                "simulation intervention; "
                "G1 did not grasp the object"
            ),
        }

    # ======================================================
    # 13. FUTURE PGAG ACTION EXECUTION
    # ======================================================

    def execute_atom(
        self,
        atom,
    ):

        raise NotImplementedError(
            f"G1 PGAG physical action "
            f"{atom.skill.value} "
            "is not yet implemented. "
            "This phase supports arm motion "
            "and block displacement tests."
        )
