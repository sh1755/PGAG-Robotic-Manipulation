"""Kinematic tabletop mock environment.

Not a physics simulator. It exists so the grammar, executor, dependency-local
repair and evaluation harness can be developed and unit-tested on a laptop with
no GPU, before Isaac Lab or the robot are in the loop. Every failure mode it
injects is controlled from the ``mock:`` section of the config:

    p_grasp_fail        grasp closes but object is not held
    p_slip_transport    object drops somewhere along the transport path
    align_noise_std     xy error (m) of the ALIGN skill
    rot_noise_deg       tilt error (deg) of the ALIGN skill
    hole_clearance      max xy error (m) for which INSERT succeeds; otherwise it jams
                        and the force guard aborts the skill
    p_disturb           per-step chance that the last placed object is knocked 4 cm
                        (tests stale-effect detection)
    obs_noise_std       Gaussian noise (m) on observed object positions
    p_collision         per-skill chance of logging a collision
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from grammar.atoms import Atom, Skill
from grammar.predicates import WorldState, quat_about_x, tilt_deg

from .base import ExecResult

DEFAULTS = dict(
    p_grasp_fail=0.08,
    p_slip_transport=0.04,
    approach_noise_std=0.003,
    align_noise_std=0.001,
    rot_noise_deg=1.0,
    hole_clearance=0.002,
    p_disturb=0.0,
    obs_noise_std=0.0,
    p_collision=0.0,
)


class MockTabletopEnv:
    def __init__(self, scene_fn: Callable[..., WorldState], cfg: Optional[Dict] = None, xy_noise: float = 0.03):
        self.scene_fn = scene_fn
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.xy_noise = xy_noise
        self.rng = np.random.default_rng(0)
        self.state: WorldState = scene_fn(self.rng, xy_noise)
        self._last_placed: Optional[str] = None

    # ------------------------------------------------------------------ Env interface
    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.state = self.scene_fn(self.rng, self.xy_noise)
        self._last_placed = None

    def observe(self) -> WorldState:
        s = self.state.copy()
        sd = self.cfg["obs_noise_std"]
        if sd > 0:
            for name, o in s.objects.items():
                if name != s.held:
                    o.pos = o.pos + self.rng.normal(0, sd, 3)
        return s

    def true_state(self) -> WorldState:
        return self.state.copy()

    def execute(self, atom: Atom) -> ExecResult:
        self._maybe_disturb()
        fn = getattr(self, f"_{atom.skill.value.lower()}")
        res: ExecResult = fn(atom)
        if self.rng.random() < self.cfg["p_collision"]:
            res.collisions += 1
        return res

    # ------------------------------------------------------------------ helpers
    def knock(self, name: str, dxy=(0.04, 0.0)) -> None:
        """Push an object sideways and let it settle (manual disturbance for tests)."""
        o = self.state[name]
        o.pos[:2] += np.asarray(dxy, float)
        self._settle(name)

    def _maybe_disturb(self) -> None:
        if self._last_placed and self.rng.random() < self.cfg["p_disturb"]:
            ang = self.rng.uniform(0, 2 * np.pi)
            self.knock(self._last_placed, 0.04 * np.array([np.cos(ang), np.sin(ang)]))
            self._last_placed = None

    def _follow(self, name: str) -> None:
        self.state.ee_pos = self.state[name].pos.copy()

    def _settle(self, name: str) -> None:
        """Drop object ``name`` straight down onto the highest support under its centre."""
        s, o = self.state, self.state[name]
        support = 0.0
        for n, other in s.objects.items():
            if n == name or not other.footprint_contains(o.pos[:2]):
                continue
            if other.kind == "container":
                support = max(support, other.bottom + 0.005)
            elif other.top <= o.bottom + 1e-3:
                support = max(support, other.top)
        o.pos[2] = support + o.size[2] / 2
        o.quat = np.array([1.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ skills
    def _approach(self, a: Atom) -> ExecResult:
        o = self.state[a.obj]
        self.state.ee_pos = o.pos + self.rng.normal(0, self.cfg["approach_noise_std"], 3)
        return ExecResult(True)

    def _grasp(self, a: Atom) -> ExecResult:
        s, o = self.state, self.state[a.obj]
        if s.held is not None:
            return ExecResult(True, info="gripper already occupied")
        if np.linalg.norm(s.ee_pos - o.pos) > 0.025:
            s.gripper_width = 0.0
            return ExecResult(True, info="closed on nothing: object not at end-effector")
        if self.rng.random() < self.cfg["p_grasp_fail"]:
            s.gripper_width = 0.0
            return ExecResult(True, info="grasp slipped")
        s.held, s.gripper_width = a.obj, float(o.size[0])
        self._follow(a.obj)
        return ExecResult(True)

    def _lift(self, a: Atom) -> ExecResult:
        s, h = self.state, a.params.get("height", 0.12)
        if s.held:
            o = s[s.held]
            o.pos[2] = h + o.size[2] / 2
            self._follow(s.held)
        else:
            s.ee_pos[2] = max(s.ee_pos[2], h)
        return ExecResult(True)

    def _transport(self, a: Atom) -> ExecResult:
        s, t = self.state, self.state[a.target]
        if s.held != a.obj:
            s.ee_pos = np.array([t.pos[0], t.pos[1], t.top + 0.10])
            return ExecResult(True, info="transport without object")
        o = s[a.obj]
        goal = np.array([t.pos[0], t.pos[1], t.top + 0.05 + o.size[2] / 2])
        if self.rng.random() < self.cfg["p_slip_transport"]:
            u = self.rng.uniform(0.2, 0.8)
            o.pos[:2] = (1 - u) * o.pos[:2] + u * goal[:2]
            s.held, s.gripper_width = None, 0.0
            self._settle(a.obj)
            s.ee_pos = goal
            return ExecResult(True, info=f"object slipped at {u:.0%} of path")
        o.pos = goal
        self._follow(a.obj)
        return ExecResult(True)

    def _align(self, a: Atom) -> ExecResult:
        s, t = self.state, self.state[a.target]
        if s.held != a.obj:
            return ExecResult(True, info="align without object")
        o = s[a.obj]
        o.pos[:2] = t.pos[:2] + self.rng.normal(0, self.cfg["align_noise_std"], 2)
        o.pos[2] = t.top + o.size[2] / 2 + 0.002
        o.quat = quat_about_x(self.rng.normal(0, self.cfg["rot_noise_deg"]))
        self._follow(a.obj)
        return ExecResult(True)

    def _insert(self, a: Atom) -> ExecResult:
        s, t = self.state, self.state[a.target]
        if s.held != a.obj:
            return ExecResult(False, info="insert without object")
        o = s[a.obj]
        err, tilt = o.xy_dist(t), tilt_deg(o.quat, t.quat)
        if err <= self.cfg["hole_clearance"] and tilt <= 3.0:
            o.pos[2] = t.top - a.params.get("depth", 0.02) + o.size[2] / 2
            self._follow(a.obj)
            return ExecResult(True)
        return ExecResult(False, guard_violations=1,
                          info=f"jam: force > {a.omega.max_force:g} N (xy err {err * 1e3:.1f} mm, tilt {tilt:.1f} deg)")

    def _release(self, a: Atom) -> ExecResult:
        s = self.state
        if s.held != a.obj:
            s.gripper_width = 0.08
            return ExecResult(True, info="release with nothing held")
        o = s[a.obj]
        s.held, s.gripper_width = None, 0.08
        t = s[a.target] if a.target else None
        if t is not None and t.kind == "hole" and o.bottom < t.top - 1e-3:
            pass  # inserted: the hole holds it in place
        else:
            self._settle(a.obj)
        self._last_placed = a.obj
        return ExecResult(True)

    def _recover(self, a: Atom) -> ExecResult:
        s = self.state
        if s.held:
            name = s.held
            s.held, s.gripper_width = None, 0.08
            self._settle(name)
        s.ee_pos = np.array([s.ee_pos[0], s.ee_pos[1], 0.30])
        return ExecResult(True, info="put down held object, retreated")
