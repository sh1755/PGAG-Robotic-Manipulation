"""Placeholder for Unitree G1 on MuJoCo.

No Unitree SDK commands are issued. Before using a real G1 or MuJoCo model,
implement connection, joint/EE state, robot kinematics, grasp controls, and
safety/stability checks against the actual installed version/configuration.

Never use this stub to command a physical humanoid.
"""

from .robot_adapter import RobotAdapter


class UnitreeG1Adapter(RobotAdapter):
    def __init__(self, config_path=None):
        self.config_path = config_path

    def reset(self, task):
        raise NotImplementedError(
            "Unitree G1 MuJoCo reset is not integrated. "
            "Use --backend mock for grammar tests."
        )

    def get_world_state(self):
        raise NotImplementedError(
            "Connect Unitree/MuJoCo state readout and map to WorldState."
        )

    def execute_atom(self, atom):
        raise NotImplementedError(
            f"Unitree motion/hand control for {atom.skill.value} "
            "has not been implemented."
        )
