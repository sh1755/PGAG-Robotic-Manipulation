"""World state and symbolic predicates for the action grammar.

Conventions
-----------
* Positions in metres, robot base frame, table surface at z = 0.
* Quaternions are (w, x, y, z).
* ``ObjectState.size`` is the full extent of the object along its own x, y, z axes.

Predicates are split into two kinds:

* **Persistent** scene relations (``on``, ``inside``, ``inserted``). Once an atom
  achieves one, it is expected to stay true until a later atom deletes it. The
  executor re-checks these before every step to catch scene changes ("stale
  effects") and invalidates the atoms that depended on them.
* **Transient** robot-relative relations (``near_ee``, ``holding``, ``lifted``,
  ``above``, ``aligned``, ``free``). These are only checked as preconditions at
  the moment an atom is about to execute.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

PERSISTENT = frozenset({"on", "inside", "inserted"})


# --------------------------------------------------------------------------- geometry helpers
def z_axis(q: np.ndarray) -> np.ndarray:
    """Body z-axis of a (w, x, y, z) quaternion, expressed in the world frame."""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def tilt_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """Angle between the z-axes of two orientations, in degrees."""
    c = float(np.clip(np.dot(z_axis(qa), z_axis(qb)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def quat_about_x(deg: float) -> np.ndarray:
    h = np.radians(deg) / 2.0
    return np.array([np.cos(h), np.sin(h), 0.0, 0.0])


# --------------------------------------------------------------------------- state
@dataclass
class ObjectState:
    pos: np.ndarray
    size: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    kind: str = "rigid"  # rigid | container | hole

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float).copy()
        self.size = np.asarray(self.size, dtype=float).copy()
        self.quat = np.asarray(self.quat, dtype=float).copy()

    @property
    def bottom(self) -> float:
        return float(self.pos[2] - self.size[2] / 2)

    @property
    def top(self) -> float:
        return float(self.pos[2] + self.size[2] / 2)

    def xy_dist(self, other: "ObjectState") -> float:
        return float(np.linalg.norm(self.pos[:2] - other.pos[:2]))

    def footprint_contains(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        return bool(
            abs(xy[0] - self.pos[0]) <= self.size[0] / 2 + margin
            and abs(xy[1] - self.pos[1]) <= self.size[1] / 2 + margin
        )


@dataclass
class WorldState:
    objects: Dict[str, ObjectState]
    ee_pos: np.ndarray
    held: Optional[str] = None
    gripper_width: float = 0.08

    def __post_init__(self):
        self.ee_pos = np.asarray(self.ee_pos, dtype=float).copy()

    def __getitem__(self, name: str) -> ObjectState:
        return self.objects[name]

    def copy(self) -> "WorldState":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- predicates
@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[str, ...] = ()
    params: Tuple[Tuple[str, float], ...] = ()

    def param(self, key: str, default: float) -> float:
        return dict(self.params).get(key, default)

    @property
    def persistent(self) -> bool:
        return self.name in PERSISTENT

    def holds(self, s: WorldState) -> bool:
        if self.name not in _EVAL:
            raise KeyError(f"No evaluator registered for predicate '{self.name}'")
        return bool(_EVAL[self.name](s, self))

    def __str__(self) -> str:
        a = ",".join(self.args)
        p = ",".join(f"{k}={v:g}" for k, v in self.params)
        return f"{self.name}({a}{'; ' + p if p else ''})"


def P(name: str, *args: str, **params: float) -> Predicate:
    """Convenience constructor: ``P("on", "b", "a", tol_xy=0.01)``."""
    return Predicate(name, tuple(args), tuple(sorted((k, float(v)) for k, v in params.items())))


def _free(s, p):
    return s.held is None


def _holding(s, p):
    return s.held == p.args[0]


def _near_ee(s, p):
    return np.linalg.norm(s.ee_pos - s[p.args[0]].pos) <= p.param("tol", 0.02)


def _lifted(s, p):
    return s[p.args[0]].bottom >= p.param("h", 0.08)


def _above(s, p):
    a, b = s[p.args[0]], s[p.args[1]]
    return a.xy_dist(b) <= p.param("tol_xy", 0.02) and a.bottom >= b.top - 1e-3


def _aligned(s, p):
    a, b = s[p.args[0]], s[p.args[1]]
    return (
        a.xy_dist(b) <= p.param("tol_xy", 0.003)
        and tilt_deg(a.quat, b.quat) <= p.param("tol_deg", 3.0)
        and a.bottom >= b.top - 0.003
    )


def _on(s, p):
    a, b = s[p.args[0]], s[p.args[1]]
    return a.xy_dist(b) <= p.param("tol_xy", 0.015) and abs(a.bottom - b.top) <= p.param("tol_z", 0.005)


def _inside(s, p):
    a, b = s[p.args[0]], s[p.args[1]]
    dx, dy = abs(a.pos[0] - b.pos[0]), abs(a.pos[1] - b.pos[1])
    return (
        dx <= (b.size[0] - a.size[0]) / 2 + 1e-3
        and dy <= (b.size[1] - a.size[1]) / 2 + 1e-3
        and a.bottom < b.top
    )


def _inserted(s, p):
    a, b = s[p.args[0]], s[p.args[1]]
    return a.xy_dist(b) <= p.param("tol_xy", 0.003) and a.bottom <= b.top - p.param("depth", 0.015)


_EVAL: Dict[str, Callable[[WorldState, Predicate], bool]] = {
    "free": _free,
    "holding": _holding,
    "near_ee": _near_ee,
    "lifted": _lifted,
    "above": _above,
    "aligned": _aligned,
    "on": _on,
    "inside": _inside,
    "inserted": _inserted,
}


def register_predicate(name: str, fn: Callable[[WorldState, Predicate], bool]) -> None:
    """Add a new predicate evaluator (e.g. when you add articulated objects)."""
    _EVAL[name] = fn
