"""Environment interface the executor talks to.

Any backend (mock, Isaac Lab, real Franka) only has to implement these four
methods. Perception plugs in behind ``observe()``: in Week 1 the mock and the
simulator return ground truth (optionally noised); later, ``observe()`` returns
the WP1 perception output with the same WorldState type.
"""
from dataclasses import dataclass
from typing import Optional, Protocol

from grammar.atoms import Atom
from grammar.predicates import WorldState


@dataclass
class ExecResult:
    ok: bool  # False only when the skill was aborted by a guard (e.g. force limit)
    collisions: int = 0
    guard_violations: int = 0
    info: str = ""


class Env(Protocol):
    def reset(self, seed: Optional[int] = None) -> None: ...

    def observe(self) -> WorldState: ...  # what the robot believes (perception)

    def true_state(self) -> WorldState: ...  # ground truth, used only for scoring

    def execute(self, atom: Atom) -> ExecResult: ...
