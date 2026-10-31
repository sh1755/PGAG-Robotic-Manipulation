
import time
import pybullet as p
import pybullet_data
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Initialize ROS 2
rclpy.init()
node = Node("franka_pybullet_controller")

pending_tasks = []

def task_callback(msg):
    task = msg.data.strip()
    if task == "pick_place":
        pending_tasks.append(task)
        node.get_logger().info("Received pick_place")
    else:
        node.get_logger().warn(
            f"Simulation not implemented for: {task}"
        )

node.create_subscription(
    String, "/pgag/task", task_callback, 10
)

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

robot = p.loadURDF(
    "franka_panda/panda.urdf",
    useFixedBase=True
)

# Create cube on the table
cube_position = [0.55, 0.0, 0.025]
cube_shape = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[0.025] * 3
)
cube_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.025] * 3,
    rgbaColor=[1, 0, 0, 1]
)
p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=cube_shape,
    baseVisualShapeIndex=cube_visual,
    basePosition=cube_position
)

arm_joints = list(range(7))
ee_link = 11

home = [0, -0.4, 0, -2.0, 0, 1.8, 0.8]

for joint, position in zip(arm_joints, home):
    p.resetJointState(robot, joint, position)

p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0.3, 0, 0.4]
)

# End-effector orientation, pointing downward
orientation = p.getQuaternionFromEuler(
    [3.14159, 0, 0]
)

def move_to(position, duration=2.0):
    """Move Franka using inverse kinematics."""
    joint_targets = p.calculateInverseKinematics(
        robot, ee_link, position,
        targetOrientation=orientation,
        maxNumIterations=100,
        residualThreshold=1e-4
    )

    steps = max(1, int(duration * 240))

    for _ in range(steps):
        p.setJointMotorControlArray(
            robot,
            arm_joints,
            p.POSITION_CONTROL,
            targetPositions=joint_targets[:7]
        )
        p.stepSimulation()
        time.sleep(1 / 240)

def execute_pick_place_demo():
    """First milestone: approach and lift motion."""
    node.get_logger().info("APPROACH")
    move_to([0.55, 0.0, 0.30])

    node.get_logger().info("MOVE NEAR OBJECT")
    move_to([0.55, 0.0, 0.17])

    node.get_logger().info("LIFT MOTION")
    move_to([0.55, 0.0, 0.35])

    node.get_logger().info("TRANSPORT MOTION")
    move_to([0.40, 0.25, 0.35])

    node.get_logger().info(
        "Motion demo completed (cube not grasped)"
    )

print("Franka ROS 2 PyBullet simulation ready")

try:
    while rclpy.ok() and p.isConnected():
        rclpy.spin_once(node, timeout_sec=0)

        if pending_tasks:
            task = pending_tasks.pop(0)
            execute_pick_place_demo()
        else:
            p.stepSimulation()
            time.sleep(1 / 240)

except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
    p.disconnect()
