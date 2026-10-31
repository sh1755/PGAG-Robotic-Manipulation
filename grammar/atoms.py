"""Grammar atoms: c_i = (skill_i, objects_i, pre_i, effect_i, Omega_i, Sigma_i).

This is the shared interface between perception (fills Sigma and the world
state), the grammar (sequences atoms), and execution (classical planner in
Track A, diffusion policy in Track B). Keep it stable; extend by adding fields
with defaults rather than changing existing ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .predicates import Predicate, WorldState


class Skill(str, Enum):
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    ALIGN = "ALIGN"
    INSERT = "INSERT"
    RELEASE = "RELEASE"
    RECOVER = "RECOVER"


@dataclass
class Guards:
    """Omega_i: hard physical constraints, checked outside any learned policy.

    Units: metres, newtons, degrees, m/s.
    """

    min_clearance: float = 0.01  # signed-distance clearance to non-target geometry
    max_force: float = 20.0  # contact wrench limit; exceeding it aborts the skill
    max_ee_speed: float = 0.25
    check_reach: bool = True  # IK reachability of the skill's target pose
    pos_tol: Optional[float] = None  # required final position accuracy
    rot_tol_deg: Optional[float] = None  # required final orientation accuracy


@dataclass
class Atom:
    """One grammar atom.

    ``effect`` are predicates the atom makes true (STRIPS add-list) and
    ``deletes`` those it makes false (delete-list). ``sigma`` maps an object
    name to its 6x6 pose covariance from perception (empty until WP1 fills it).
    """

    id: str
    skill: Skill
    objects: Tuple[str, ...]
    pre: List[Predicate] = field(default_factory=list)
    effect: List[Predicate] = field(default_factory=list)
    deletes: List[Predicate] = field(default_factory=list)
    omega: Guards = field(default_factory=Guards)
    sigma: Dict[str, np.ndarray] = field(default_factory=dict)
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def obj(self) -> str:
        return self.objects[0]

    @property
    def target(self) -> Optional[str]:
        return self.objects[1] if len(self.objects) > 1 else None

    def failed_pre(self, s: WorldState) -> List[Predicate]:
        return [p for p in self.pre if not p.holds(s)]

    def failed_effect(self, s: WorldState) -> List[Predicate]:
        return [p for p in self.effect if not p.holds(s)]

    def __str__(self) -> str:
        return f"{self.id}: {self.skill.value}({', '.join(self.objects)})"


def recover_atom(obj: str) -> Atom:
    """Generic recovery: put down whatever is held, move to a safe pose."""
    return Atom(id=f"recover_{obj}", skill=Skill.RECOVER, objects=(obj,))
