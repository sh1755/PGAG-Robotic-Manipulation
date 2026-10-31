"""Initial scenes for the three tasks (used by the mock env; mirror these in Isaac Lab).

Object xy positions are jittered by +/- ``xy_noise`` metres per episode.
"""
from typing import Callable, Dict

import numpy as np

from grammar.predicates import ObjectState, WorldState

HOME_EE = np.array([0.30, 0.0, 0.40])


def _j(rng: np.random.Generator, xy_noise: float) -> np.ndarray:
    return np.append(rng.uniform(-xy_noise, xy_noise, 2), 0.0)


def scene_pick_place(rng, xy_noise=0.03) -> WorldState:
    return WorldState(
        objects={
            "cube": ObjectState(pos=np.array([0.50, 0.00, 0.02]) + _j(rng, xy_noise), size=[0.04] * 3),
            "bin": ObjectState(pos=np.array([0.40, 0.30, 0.03]) + _j(rng, xy_noise / 2),
                               size=[0.20, 0.20, 0.06], kind="container"),
        },
        ee_pos=HOME_EE,
    )


def scene_stack(rng, xy_noise=0.03) -> WorldState:
    s = 0.05
    return WorldState(
        objects={
            "a": ObjectState(pos=np.array([0.55, 0.00, s / 2]) + _j(rng, xy_noise), size=[s] * 3),
            "b": ObjectState(pos=np.array([0.45, -0.18, s / 2]) + _j(rng, xy_noise), size=[s] * 3),
            "c": ObjectState(pos=np.array([0.45, 0.18, s / 2]) + _j(rng, xy_noise), size=[s] * 3),
        },
        ee_pos=HOME_EE,
    )


def scene_peg_insert(rng, xy_noise=0.03) -> WorldState:
    return WorldState(
        objects={
            "peg": ObjectState(pos=np.array([0.50, -0.12, 0.04]) + _j(rng, xy_noise), size=[0.02, 0.02, 0.08]),
            "hole": ObjectState(pos=np.array([0.50, 0.15, 0.025]) + _j(rng, xy_noise),
                                size=[0.10, 0.10, 0.05], kind="hole"),
        },
        ee_pos=HOME_EE,
    )


SCENES: Dict[str, Callable[..., WorldState]] = {
    "pick_place": scene_pick_place,
    "stack": scene_stack,
    "peg_insert": scene_peg_insert,
}
