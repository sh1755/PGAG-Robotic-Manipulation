
import time
import math
import pybullet as p
import pybullet_data

# Start the 3D simulator
client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load floor and Franka Panda
p.loadURDF("plane.urdf")

robot = p.loadURDF(
    "franka_panda/panda.urdf",
    [0, 0, 0],
    useFixedBase=True
)

# Configure the camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.6,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# Move the Panda's seven arm joints
arm_joints = list(range(7))

initial_positions = [
    0, -0.4, 0, -2.0, 0, 1.8, 0.8
]

for joint, angle in zip(arm_joints, initial_positions):
    p.resetJointState(robot, joint, angle)

print("Franka Panda simulation running")

try:
    t = 0
    while p.isConnected():
        targets = initial_positions.copy()

        # Demonstration movement of joint 1
        targets[0] = 0.3 * math.sin(t)

        p.setJointMotorControlArray(
            robot,
            arm_joints,
            p.POSITION_CONTROL,
            targetPositions=targets
        )

        p.stepSimulation()
        time.sleep(1 / 240)
        t += 1 / 240

except KeyboardInterrupt:
    pass

finally:
    p.disconnect()
