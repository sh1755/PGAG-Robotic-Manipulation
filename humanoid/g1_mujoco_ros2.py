"""Optional ROS 2 interface to the real G1 MuJoCo *smoke tests*.

Topic isolation prevents accidental triggering of Franka or symbolic mock nodes.
Inputs:  /pgag/g1_mujoco/task   std_msgs/String  ('scene','arm_demo','disturb')
Outputs: /pgag/g1_mujoco/result std_msgs/String  (JSON)

DOES NOT send Unitree SDK commands or execute grasping/stacking.
"""
import argparse
import json
from collections import deque

from .g1_mujoco_adapter import G1MujocoAdapter, default_model_path


def main():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model-path", default=str(default_model_path()))
    args = parser.parse_args()

    class G1PhysicsNode(Node):
        def __init__(self, sim):
            super().__init__("pgag_g1_mujoco_test")
            self.sim = sim
            self.pending = deque()
            self.create_subscription(String, "/pgag/g1_mujoco/task", self.pending.append, 10)
            self.results = self.create_publisher(String, "/pgag/g1_mujoco/result", 10)
            self.create_timer(0.02, self.tick)
            self.get_logger().info("G1 FIXED-BASE MUJOCO test ready: scene, arm_demo, disturb")

        def tick(self):
            if not self.pending:
                self.sim.step(10)
                return
            task = self.pending.popleft().data.strip()
            try:
                if task == "arm_demo":
                    result = self.sim.arm_demo()
                elif task == "disturb":
                    result = self.sim.disturb_block_b()
                elif task == "scene":
                    result = {
                        "task": "scene", "success": True,
                        "on_b_a": self.sim.on_b_on_a(),
                        "wrist_xyz": self.sim.wrist_xyz().tolist(),
                        "physics_engine": "mujoco",
                    }
                else:
                    result = {"success": False, "error": f"Unsupported task: {task}"}
            except Exception as exc:
                result = {"task": task, "success": False, "error": str(exc)}
            self.results.publish(String(data=json.dumps(result)))
            self.get_logger().info(json.dumps(result))

    rclpy.init()
    try:
        with G1MujocoAdapter(args.model_path, gui=args.gui) as sim:
            node = G1PhysicsNode(sim)
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            finally:
                node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
