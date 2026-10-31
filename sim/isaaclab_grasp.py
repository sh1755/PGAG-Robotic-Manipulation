"""Week 1 Isaac Lab scene: Franka + cube, scripted APPROACH -> GRASP -> LIFT atoms.

Written for Isaac Lab 2.x (package name `isaaclab`). Run from your Isaac Lab
install, pointing at this file:

    cd ~/IsaacLab
    ./isaaclab.sh -p ~/pgag/sim/isaaclab_grasp.py --episodes 10              # with GUI
    ./isaaclab.sh -p ~/pgag/sim/isaaclab_grasp.py --episodes 50 --headless   # batch

Each episode randomises the cube position, runs the three atoms with a
differential-IK controller, checks the atom effects with the same thresholds
as grammar/predicates.py (near_ee, holding, lifted), and appends a row to a CSV
with the same columns as eval/run.py.

This is the bridge between the grammar and the simulator. In Week 2 the phase
machine below is replaced by `Env.execute(atom)` so the full executor drives it.
"""
import argparse
import csv
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted APPROACH-GRASP-LIFT on Franka in Isaac Lab.")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--xy_noise", type=float, default=0.05, help="cube position jitter (m)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="runs/isaaclab_grasp.csv")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that need the simulator running must come after AppLauncher.
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

CUBE = 0.04  # edge length (m), same as envs/scenes.py
HAND_TO_TCP = 0.1034  # panda_hand origin to fingertip centre (m)
DOWN = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z): gripper pointing down

# Thresholds mirror grammar/predicates.py
NEAR_TOL = 0.02
LIFT_H = 0.08


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE, CUBE, CUBE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, CUBE / 2)),
    )


def ee_pose_base(robot, ee_idx):
    """End-effector pose in the robot base frame (positions/quats as tensors)."""
    ee_w = robot.data.body_state_w[:, ee_idx, 0:7]
    root_w = robot.data.root_state_w[:, 0:7]
    return subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view([1.6, 1.2, 1.0], [0.4, 0.0, 0.1])
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot, cube = scene["robot"], scene["cube"]
    arm = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm.resolve(scene)
    fingers = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    fingers.resolve(scene)
    ee_idx = arm.body_ids[0]
    jac_idx = ee_idx - 1  # fixed-base articulation: the Jacobian omits the root body

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=sim.device)
    down = torch.tensor([DOWN], device=sim.device)
    gen = torch.Generator(device="cpu").manual_seed(args_cli.seed)

    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args_cli.out)
    f = open(args_cli.out, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["env", "task", "episode", "seed", "success", "termination", "steps", "approach_ok",
                    "grasp_ok", "lift_ok", "cube_z_final", "ee_cube_err_mm", "episode_time_s"])

    # Phases: (name, duration in sim steps)
    phases = [("pregrasp", 150), ("approach", 120), ("grasp", 60), ("lift", 150)]
    dt = sim.get_physics_dt()

    for ep in range(args_cli.episodes):
        t_wall = time.time()
        # ---------------- reset robot and cube
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        robot.reset()
        jitter = (torch.rand(2, generator=gen) * 2 - 1) * args_cli.xy_noise
        cube_state = cube.data.default_root_state.clone()
        cube_state[:, 0:2] += jitter.to(sim.device)
        cube_state[:, 0:3] += scene.env_origins
        cube.write_root_state_to_sim(cube_state)
        cube.reset()
        ik.reset()
        scene.write_data_to_sim()
        for _ in range(30):  # let things settle
            sim.step()
            scene.update(dt)

        cube_b = cube.data.root_state_w[:, 0:3] - scene.env_origins  # robot base sits at env origin
        result = {"approach_ok": 0, "grasp_ok": 0, "lift_ok": 0}

        for name, n_steps in phases:
            target = cube_b.clone()
            if name == "pregrasp":
                target[:, 2] += HAND_TO_TCP + 0.10
                grip = 0.04
            elif name == "approach":
                target[:, 2] += HAND_TO_TCP
                grip = 0.04
            elif name == "grasp":
                target[:, 2] += HAND_TO_TCP
                grip = 0.0
            else:  # lift
                target[:, 2] += HAND_TO_TCP + 0.12
                grip = 0.0
            ik.set_command(torch.cat([target, down], dim=-1))

            for _ in range(n_steps):
                ee_pos, ee_quat = ee_pose_base(robot, ee_idx)
                jac = robot.root_physx_view.get_jacobians()[:, jac_idx, :, arm.joint_ids]
                q_des = ik.compute(ee_pos, ee_quat, jac, robot.data.joint_pos[:, arm.joint_ids])
                robot.set_joint_position_target(q_des, joint_ids=arm.joint_ids)
                robot.set_joint_position_target(torch.full((1, len(fingers.joint_ids)), grip, device=sim.device),
                                                joint_ids=fingers.joint_ids)
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

            # ---------------- effect checks (same semantics as grammar/predicates.py)
            ee_pos, _ = ee_pose_base(robot, ee_idx)
            tcp = ee_pos.clone()
            tcp[:, 2] -= HAND_TO_TCP
            cube_now = cube.data.root_state_w[:, 0:3] - scene.env_origins
            err = torch.linalg.norm(tcp - cube_now, dim=-1).item()
            if name == "approach":
                result["approach_ok"] = int(err <= NEAR_TOL)
            elif name == "grasp":
                width = robot.data.joint_pos[:, fingers.joint_ids].sum().item()
                result["grasp_ok"] = int(width > 0.6 * CUBE)  # fingers stopped on the cube, not closed
            elif name == "lift":
                cube_bottom = cube_now[0, 2].item() - CUBE / 2
                result["lift_ok"] = int(cube_bottom >= LIFT_H and err <= 0.03)

        z_final = (cube.data.root_state_w[0, 2] - scene.env_origins[0, 2]).item()
        success = all(result.values())
        w.writerow(["isaaclab", "grasp_lift", ep, args_cli.seed, int(success),
                    "program_complete" if success else "effect_failed", 3,
                    result["approach_ok"], result["grasp_ok"], result["lift_ok"],
                    round(z_final, 4), round(err * 1e3, 2), round(time.time() - t_wall, 2)])
        f.flush()
        print(f"episode {ep:>3}: approach={result['approach_ok']} grasp={result['grasp_ok']} "
              f"lift={result['lift_ok']}  cube z={z_final:.3f} m  -> {'SUCCESS' if success else 'FAIL'}")

    f.close()
    print(f"Results appended to {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
