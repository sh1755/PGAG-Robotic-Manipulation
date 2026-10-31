# Unitree G1 – MuJoCo Physics Phase 1 (PGAG)

This add-on is an **actual Unitree G1 29-DoF MuJoCo physics smoke test**,
not a symbolic mock. It reads the official MJCF model, dynamically creates a
physical tabletop and three blocks, and sends PD motor torques to the real
simulated G1 arm joints. The pelvis is **fixed to the world** for this first
manipulation-oriented test (not dynamically balancing or walking).

**Scope:** load model, inspect scene, drive right shoulder and elbow with
measured joint-position feedback, read wrist pose, physically move a block
and verify the PGAG `on(b,a)` relation.

**Not implemented:** articulated fingers, contact-based grasp, full PGAG
pick_place/stack atoms on G1, whole-body balance, recovery by G1, DDS or
real Unitree hardware. A standard `g1_29dof.xml` has no actuated finger joints;
choose a hand-equipped G1 model before implementing grasping.

## Install Ubuntu 22.04

From your existing project root:

```bash
cd ~/Desktop/Robotics_Research
# If not already cloned:
git clone https://github.com/unitreerobotics/unitree_mujoco.git

cd ~/Desktop/Robotics_Research/pgag_week1/pgag
source ../.venv/bin/activate
python -m pip install 'mujoco>=3.3,<4'
```

The model should be at:
`~/Desktop/Robotics_Research/unitree_mujoco/unitree_robots/g1/scene_29dof.xml`.

## Run

```bash
cd ~/Desktop/Robotics_Research/pgag_week1/pgag
source ../.venv/bin/activate
python -m humanoid.g1_mujoco_probe --once scene --headless
python -m humanoid.g1_mujoco_probe --once arm_demo --gui --realtime
python -m humanoid.g1_mujoco_probe --once disturb --gui --realtime
```

If the `--gui` viewer cannot open, check graphical desktop support and use
`--headless` instead. The `--realtime` option intentionally slows the simulation.

If you cloned Unitree elsewhere, provide the model location explicitly:

```bash
python -m humanoid.g1_mujoco_probe --once arm_demo \
  --model-path /absolute/path/to/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --gui
```

## Optional ROS 2 interface (same physics engine)

```bash
# Terminal 1 (stop Franka and the older mock G1 ROS bridge first)
source /opt/ros/humble/setup.bash
source ../.venv/bin/activate
python -m humanoid.g1_mujoco_ros2 --gui

# Terminal 2
source /opt/ros/humble/setup.bash
ros2 topic echo /pgag/g1_mujoco/result

# Terminal 3
source /opt/ros/humble/setup.bash
ros2 topic pub --once /pgag/g1_mujoco/task std_msgs/msg/String "{data: 'arm_demo'}"
ros2 topic pub --once /pgag/g1_mujoco/task std_msgs/msg/String "{data: 'disturb'}"
```

These are **not** the old `/pgag/task` command topics by design. The physical
G1 adapter does not implement pick-and-place. The original `humanoid/mock_g1_adapter.py`
and its five tests remain unchanged and work separately.

## Save changes

```bash
git add humanoid/g1_mujoco_adapter.py humanoid/g1_mujoco_probe.py \
        humanoid/g1_mujoco_ros2.py humanoid/G1_MUJOCO_SETUP.md
git commit -m "Add G1 MuJoCo fixed-base physics and arm control prototype"
git pull --rebase origin main
git push origin main
```

Do not connect this exploratory adapter to physical hardware. Full safety,
contact grasping and balance require their own controllers and testing.
