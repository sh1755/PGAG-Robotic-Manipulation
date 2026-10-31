"""Grasp-related safety checks for an eventual Unitree MuJoCo implementation.

This module intentionally does NOT command the Unitree robot.
"""

import math


def tcp_object_distance(world, object_name):
    """Geometric proximity only; not evidence of physical finger contact."""
    return math.dist(world.ee_pos, world.objects[object_name].pos)


def check_grasp_proximity(world, object_name, max_distance_m=0.03):
    distance = tcp_object_distance(world, object_name)
    if distance > max_distance_m:
        raise RuntimeError(
            f"Grasp approach too far from {object_name}: {distance:.3f} m"
        )
    return distance
