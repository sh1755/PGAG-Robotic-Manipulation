"""Standalone ROS 2 adapter for PGAG's mock task planner.
Run this file from the original pgag/ directory with ROS 2 Humble sourced.
Does not command a physical robot.
"""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from grammar.programs import TASKS
from envs.scenes import SCENES
from envs.mock_env import MockTabletopEnv
from execution.executor import GrammarExecutor, ExecutorConfig


class PGAGBridge(Node):
    def __init__(self):
        super().__init__('pgag_bridge')
        self.plan_pub = self.create_publisher(String, '/pgag/plan', 10)
        self.result_pub = self.create_publisher(String, '/pgag/result', 10)
        self.task_sub = self.create_subscription(String, '/pgag/task', self.on_task, 10)
        self.get_logger().info('Ready: publish pick_place, stack, or peg_insert to /pgag/task')

    def on_task(self, message):
        task = message.data.strip()
        try:
            if task not in TASKS:
                raise ValueError(f'Unknown task {task}. Choose: {", ".join(TASKS)}')
            program = TASKS[task]()
            issues = program.validate()
            if issues:
                raise ValueError('; '.join(issues))
            plan = {
                'task': task,
                'actions': [
                    {'id': action.id, 'skill': action.skill.value,
                     'objects': list(action.objects)}
                    for action in program.atoms
                ],
            }
            self.plan_pub.publish(String(data=json.dumps(plan)))
            env = MockTabletopEnv(SCENES[task])
            env.reset(seed=0)
            outcome = GrammarExecutor(program, env, ExecutorConfig()).run()
            result = {
                'task': task, 'environment': 'mock',
                'success': bool(outcome.success), 'steps': outcome.steps,
                'local_repairs': outcome.local_repairs,
                'full_replans': outcome.full_replans,
            }
            self.result_pub.publish(String(data=json.dumps(result)))
            self.get_logger().info(f'{task}: success={outcome.success}')
        except Exception as error:
            self.get_logger().error(str(error))
            self.result_pub.publish(String(data=json.dumps({
                'task': task, 'success': False, 'error': str(error)
            })))


def main():
    rclpy.init()
    node = PGAGBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
