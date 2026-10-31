"""No-physics reference adapter. This is NOT a Unitree control simulator.

Allows verifying the grammar program, predicates, and local recovery selection
on a laptop without a Unitree SDK, MuJoCo, ROS 2, or robot hardware.
"""

import numpy as np

from grammar.atoms import Skill
from grammar.predicates import ObjectState, WorldState

from .robot_adapter import RobotAdapter


SIZE = 0.05

STARTS = {
    "a": [0.46, 0.18, 0.025],
    "b": [0.52, -0.16, 0.025],
    "c": [0.61, -0.02, 0.025],
    "cube": [0.55, 0.00, 0.025],
}


class MockG1Adapter(RobotAdapter):
    """An idealised tabletop state machine, not dynamics or a humanoid model."""

    def __init__(self):
        self.world = None
        self.task = None

    def reset(self, task):
        if task not in ("stack", "pick_place"):
            raise ValueError(f"Unsupported mock task: {task}")
        self.task = task
        names = ("a", "b", "c") if task == "stack" else ("cube",)
        objects = {
            name: ObjectState(pos=STARTS[name], size=[SIZE] * 3)
            for name in names
        }
        if task == "pick_place":
            objects["bin"] = ObjectState(
                pos=[0.40, 0.25, 0.05],
                size=[0.14, 0.14, 0.10],
                kind="container",
            )
        self.world = WorldState(
            objects=objects,
            ee_pos=np.array([0.20, 0.00, 0.40]),
            held=None,
            gripper_width=0.08,
        )

    def get_world_state(self):
        if self.world is None:
            raise RuntimeError("Call reset() before reading world state.")
        return self.world.copy()

    def execute_atom(self, atom):
        if self.world is None:
            raise RuntimeError("Call reset() before executing atoms.")

        s = self.world
        name = atom.obj
        obj = s[name]
        target = s[atom.target] if atom.target else None

        if atom.skill == Skill.APPROACH:
            s.ee_pos = obj.pos.copy()

        elif atom.skill == Skill.GRASP:
            if s.held is not None:
                raise RuntimeError("Mock hand is occupied.")
            s.held = name
            s.gripper_width = 0.04

        elif atom.skill == Skill.LIFT:
            if s.held != name:
                raise RuntimeError(f"Cannot lift unheld object: {name}")
            obj.pos[2] = max(obj.pos[2], 0.20)
            s.ee_pos = obj.pos.copy()

        elif atom.skill == Skill.TRANSPORT:
            if s.held != name or target is None:
                raise RuntimeError("Invalid transport request.")
            obj.pos = np.array([
                target.pos[0],
                target.pos[1],
                target.top + SIZE / 2 + 0.08,
            ])
            s.ee_pos = obj.pos.copy()

        elif atom.skill == Skill.ALIGN:
            if s.held != name or target is None:
                raise RuntimeError("Invalid align request.")
            obj.pos = np.array([
                target.pos[0],
                target.pos[1],
                target.top + SIZE / 2 + 0.015,
            ])
            s.ee_pos = obj.pos.copy()

        elif atom.skill == Skill.RELEASE:
            if s.held != name or target is None:
                raise RuntimeError("Invalid release request.")
            if target.kind == "container":
                obj.pos = np.array([target.pos[0], target.pos[1], SIZE / 2])
            else:
                obj.pos = np.array([
                    target.pos[0],
                    target.pos[1],
                    target.top + SIZE / 2,
                ])
            s.held = None
            s.gripper_width = 0.08
            s.ee_pos = obj.pos + np.array([0.0, 0.0, 0.12])

        else:
            raise NotImplementedError(f"Mock does not support {atom.skill.value}")

    def inject_displacement(self, obj):
        if self.world is None or self.world.held is not None:
            raise RuntimeError("Can disturb only released blocks.")
        if obj not in self.world.objects:
            raise KeyError(obj)
        self.world[obj].pos = np.array(STARTS[obj], dtype=float)
