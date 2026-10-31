# PGAG → Unitree G1 Humanoid Research Scaffold

This folder is an **add-on** for the existing PGAG repository. It does **not**
replace or modify `grammar/`, `execution/`, or `franka_ros2_sim*.py`.

## Status

- **Working:** symbolic/mock PGAG action execution for pick_place and stack;
  predicate checks; scripted displacement of Block 2 and targeted replay.
- **Not implemented:** G1 MuJoCo connection, joint index mapping, IK, hand
  drivers, real robot grasping, balancing, collision checking, ROS 2 to
  Unitree SDK control and hardware deployment.
- Mock successes are **not physical manipulation results**.

## Files

| File | Purpose |
|---|---|
| `robot_adapter.py` | Robot-agnostic adapter interface |
| `mock_g1_adapter.py` | Mock state-machine adapter to test integration |
| `unitree_g1_adapter.py` | Explicit stub for later Unitree MuJoCo integration |
| `pgag_unitree_sim.py` | Mock PGAG task runner, fault-injection experiment |
| `g1_state_monitor.py` | Existing PGAG predicate checks |
| `g1_grasp_controller.py` | Grasp proximity helper; no hardware commands |
| `g1_failure_recovery.py` | Task-specific local replay of B-on-A atoms |
| `g1_ros2_bridge.py` | Optional ROS 2 interface for mock experiments |
| `configs/g1_stationary.yaml` | Illustrative G1 research settings |
| `tests/test_scaffold.py` | Five quick mock tests |

## Run (from the PGAG repo root)

```bash
cd ~/Desktop/Robotics_Research/pgag_week1/pgag
source ../.venv/bin/activate

python -m humanoid.pgag_unitree_sim --task pick_place
python -m humanoid.pgag_unitree_sim --task stack
python -m humanoid.pgag_unitree_sim --task stack --disturb
python -m humanoid.pgag_unitree_sim --task stack --disturb --no-repair

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q humanoid/tests
```

The last fault-injection command exits with code 1 by design because repair
is disabled.

Optional mock-only ROS 2 interface:

```bash
source /opt/ros/humble/setup.bash
source ../.venv/bin/activate
python -m humanoid.g1_ros2_bridge
# Other terminal:
ros2 topic pub --once /pgag/humanoid/task std_msgs/msg/String "{data: 'stack_disturb'}"
ros2 topic echo /pgag/humanoid/result
```

## Next integration steps

1. Choose exact G1 robot model, hand hardware/configuration and MuJoCo version.
2. Connect a Unitree MuJoCo client; map measured hand and object states to
   the shared `WorldState` with correct reference frames.
3. Implement `UnitreeG1Adapter.execute_atom` with upper-body IK, a wrist/hand
   controller, collision checking and explicit support/balance checks.
4. Validate reach, finger-object contacts, stable release and disturbances in
   MuJoCo before claiming physical results.
5. Replace the task-specific replay with the project `GrammarExecutor`
   dependency-local recovery mechanism.
6. Record reproducible comparisons: task success, task time, number of
   re-executed atoms, recovery time and intervention count.

Do not connect this scaffold to physical G1 hardware without validated vendor
interfaces, safety interlocks and supervised testing.
