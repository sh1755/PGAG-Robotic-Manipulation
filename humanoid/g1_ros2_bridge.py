"""Optional ROS 2 wrapper for the MOCK PGAG humanoid demo.

Uses separate topics to avoid simultaneously triggering the Franka nodes:
    /pgag/humanoid/task   std_msgs/String (stack, stack_disturb, pick_place)
    /pgag/humanoid/result std_msgs/String (JSON result)

This does not connect to Unitree or MuJoCo.
"""

import json
from collections import deque

from .pgag_unitree_sim import execute_pgag


def main():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class HumanoidMockNode(Node):
        def __init__(self):
            super().__init__("pgag_humanoid_mock")
            self.pending = deque()
            self.create_subscription(
                String, "/pgag/humanoid/task",
                self.pending.append, 10
            )
            self.result_pub = self.create_publisher(
                String, "/pgag/humanoid/result", 10
            )
            self.create_timer(0.1, self.tick)
            self.get_logger().info("MOCK humanoid bridge ready")

        def tick(self):
            if not self.pending:
                return
            name = self.pending.popleft().data.strip()
            if name not in ("stack", "stack_disturb", "pick_place"):
                result = {"success": False, "error": f"Unsupported task: {name}"}
            else:
                result = execute_pgag(
                    task="stack" if name == "stack_disturb" else name,
                    disturb=name == "stack_disturb",
                )
            self.result_pub.publish(String(data=json.dumps(result)))
            self.get_logger().info(
                f"Task {name}: success={result['success']}"
            )

    rclpy.init()
    node = HumanoidMockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
