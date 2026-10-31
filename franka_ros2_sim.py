
import time
import math

import pybullet as p
import pybullet_data
import rclpy

from rclpy.node import Node
from std_msgs.msg import String


# ==========================================================
# 1. ROS 2 INITIALISATION
# ==========================================================

rclpy.init()

node = Node("franka_grasp_controller")
tasks = []


def callback(msg):
    task = msg.data.strip()

    if task == "pick_place":
        tasks.append(task)
        node.get_logger().info("Task received: pick_place")
    else:
        node.get_logger().warning(
            f"Unsupported task: {task}"
        )


node.create_subscription(
    String,
    "/pgag/task",
    callback,
    10
)


# ==========================================================
# 2. PYBULLET INITIALISATION
# ==========================================================

client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)

p.loadURDF("plane.urdf")

robot = p.loadURDF(
    "franka_panda/panda.urdf",
    useFixedBase=True
)


# ==========================================================
# 3. ROBOT CONFIGURATION
# ==========================================================

arm = list(range(7))
fingers = [9, 10]
ee = 11

home = [0, -0.4, 0, -2, 0, 1.8, 0.8]

cube_start = [0.55, 0, 0.025]
cube_target = [0.40, 0.25, 0.025]

cube_size = 0.05

orientation = p.getQuaternionFromEuler(
    [math.pi, 0, 0]
)


# ==========================================================
# 4. CREATE CUBE
# ==========================================================

collision_shape = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[cube_size / 2] * 3
)

visual_shape = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[cube_size / 2] * 3,
    rgbaColor=[1, 0, 0, 1]
)

cube = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=cube_start
)

p.changeDynamics(
    cube,
    -1,
    lateralFriction=1.0
)


# ==========================================================
# 5. CAMERA
# ==========================================================

p.resetDebugVisualizerCamera(
    cameraDistance=1.1,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0.45, 0, 0.20]
)


# ==========================================================
# 6. SIMULATION FUNCTIONS
# ==========================================================

def step(n=240):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1 / 240)


def get_ee_position():
    state = p.getLinkState(
        robot,
        ee,
        computeForwardKinematics=True
    )
    return state[4]


def get_cube_position():
    position, _ = p.getBasePositionAndOrientation(cube)
    return position


# ==========================================================
# 7. GRIPPER CONTROL
# ==========================================================

def gripper(width):
    width = max(0.0, min(0.04, width))

    for joint in fingers:
        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=width,
            force=40
        )

    step(180)


# ==========================================================
# 8. MOVE ROBOT USING INVERSE KINEMATICS
# ==========================================================

def move(position, duration=480):

    q = p.calculateInverseKinematics(
        robot,
        ee,
        position,
        targetOrientation=orientation,
        maxNumIterations=200,
        residualThreshold=1e-5
    )

    for _ in range(duration):

        p.setJointMotorControlArray(
            robot,
            arm,
            p.POSITION_CONTROL,
            targetPositions=q[:7],
            forces=[87, 87, 87, 87, 12, 12, 12]
        )

        p.stepSimulation()
        time.sleep(1 / 240)

    actual = get_ee_position()
    error = math.dist(actual, position)

    node.get_logger().info(
        f"IK target error: {error:.3f} m"
    )

    return error


# ==========================================================
# 9. RESET SCENE
# ==========================================================

def reset_scene():

    node.get_logger().info("RESET SCENE")

    p.resetBasePositionAndOrientation(
        cube,
        cube_start,
        [0, 0, 0, 1]
    )

    p.resetBaseVelocity(
        cube,
        linearVelocity=[0, 0, 0],
        angularVelocity=[0, 0, 0]
    )

    for joint, angle in zip(arm, home):
        p.resetJointState(
            robot,
            joint,
            angle,
            targetVelocity=0
        )

        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=angle
        )

    for joint in fingers:
        p.resetJointState(robot, joint, 0.04)
        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=40
        )

    step(120)

    node.get_logger().info("Scene reset complete")


# ==========================================================
# 10. GRASP VERIFICATION
# ==========================================================

def grasp_constraint():

    ee_pos = get_ee_position()
    cube_pos = get_cube_position()

    distance = math.dist(ee_pos, cube_pos)

    node.get_logger().info(
        f"End effector: {ee_pos}"
    )

    node.get_logger().info(
        f"Cube position: {cube_pos}"
    )

    node.get_logger().info(
        f"Grasp distance: {distance:.3f} m"
    )

    # Require close alignment before attaching cube
    if distance > 0.035:
        raise RuntimeError(
            f"Gripper not aligned: {distance:.3f} m"
        )

    link = p.getLinkState(
        robot,
        ee,
        computeForwardKinematics=True
    )

    ee_orn = link[5]

    cube_position, cube_orn = (
        p.getBasePositionAndOrientation(cube)
    )

    inv_pos, inv_orn = p.invertTransform(
        ee_pos,
        ee_orn
    )

    local_pos, local_orn = p.multiplyTransforms(
        inv_pos,
        inv_orn,
        cube_position,
        cube_orn
    )

    attachment = p.createConstraint(
        robot,
        ee,
        cube,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        local_pos,
        [0, 0, 0],
        parentFrameOrientation=local_orn
    )

    node.get_logger().info(
        "Cube attached to gripper"
    )

    return attachment


# ==========================================================
# 11. PICK AND PLACE
# ==========================================================

def pick_place():

    attachment = None

    try:

        reset_scene()

        # Get actual cube position
        cube_pos = get_cube_position()

        x = cube_pos[0]
        y = cube_pos[1]
        z = cube_pos[2]

        # APPROACH
        node.get_logger().info("APPROACH")

        gripper(0.04)

        approach_position = [
            x,
            y,
            z + 0.25
        ]

        move(approach_position)

        # GRASP
        node.get_logger().info("GRASP")

        grasp_position = [
            x,
            y,
            z + 0.005
        ]

        grasp_error = move(
            grasp_position,
            duration=600
        )

        if grasp_error > 0.035:
            raise RuntimeError(
                f"IK grasp error too large: {grasp_error:.3f} m"
            )

        gripper(0.015)

        attachment = grasp_constraint()

        # LIFT
        node.get_logger().info("LIFT")

        move([
            x,
            y,
            z + 0.30
        ])

        # TRANSPORT
        node.get_logger().info("TRANSPORT")

        move([
            cube_target[0],
            cube_target[1],
            0.35
        ])

        # LOWER
        node.get_logger().info("LOWER")

        move([
            cube_target[0],
            cube_target[1],
            cube_target[2] + 0.005
        ])

        # RELEASE
        node.get_logger().info("RELEASE")

        p.removeConstraint(attachment)
        attachment = None

        gripper(0.04)

        # Retreat after release
        move([
            cube_target[0],
            cube_target[1],
            0.30
        ])

        step(240)

        # Final placement evaluation
        final_pos = get_cube_position()

        placement_error = math.dist(
            final_pos[:2],
            cube_target[:2]
        )

        node.get_logger().info(
            f"Final cube position: {final_pos}"
        )

        node.get_logger().info(
            f"Placement XY error: {placement_error:.3f} m"
        )

        if placement_error <= 0.03:
            node.get_logger().info(
                "PICK AND PLACE SUCCESS"
            )
        else:
            node.get_logger().warning(
                "Placement outside tolerance"
            )

    except Exception as exc:

        node.get_logger().error(
            f"Task failed: {exc}"
        )

    finally:

        if attachment is not None:
            p.removeConstraint(attachment)


# ==========================================================
# 12. MAIN ROS 2 LOOP
# ==========================================================

node.get_logger().info(
    "Franka Panda ROS 2 simulation ready"
)

try:

    while rclpy.ok() and p.isConnected():

        rclpy.spin_once(
            node,
            timeout_sec=0
        )

        if tasks:

            task = tasks.pop(0)

            if task == "pick_place":
                pick_place()

        else:

            p.stepSimulation()
            time.sleep(1 / 240)

except KeyboardInterrupt:

    node.get_logger().info(
        "Stopping Franka simulation"
    )

finally:

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    if p.isConnected():
        p.disconnect()
