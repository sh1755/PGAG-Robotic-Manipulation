"""Grammar executor with dependency-local repair (Track A skeleton).

Loop, one decision per iteration:

1. Observe the scene.
2. Stale check: did any *persistent* effect of a completed atom stop holding
   (e.g. a placed block was knocked)? -> invalidate the closure of those atoms.
3. Precondition check for the next pending atom. Missing predicate ->
   find the atom that produced it and invalidate that atom's closure.
4. Execute the atom (classical controller now; diffusion policy in Track B).
5. Effect check. On repeated failure with preconditions still satisfied, back
   off one dependency level (re-run the producers of its geometric
   preconditions, e.g. re-ALIGN after an INSERT jam).
6. If local repair cannot help (retry budget exhausted, predicate supplied by
   the initial state), fall back to a full replan.

``local_repair=False`` turns every failure into a full replan: that is the
"full re-grounding" baseline for the selective-vs-full ablation.

Hook for WP4: ``risk_fn(atom, state) -> probability of guard violation``. If it
exceeds ``risk_threshold`` the atom is not executed and its geometric producers
are re-run first. It is None (disabled) in Week 1.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from envs.base import Env
from grammar.atoms import Atom, recover_atom
from grammar.dependency import closure, stale_atoms
from grammar.programs import INITIAL_ID, TaskProgram

GRIPPER_STATE = {"holding", "free"}


@dataclass
class ExecutorConfig:
    max_steps: int = 60  # skill executions per episode
    max_atom_retries: int = 3
    max_full_replans: int = 2
    local_repair: bool = True
    risk_threshold: float = 0.5


@dataclass
class EpisodeResult:
    success: bool
    termination: str
    steps: int
    local_repairs: int
    full_replans: int
    stale_detections: int
    collisions: int
    guard_violations: int
    goal_xy_err_mm: float
    decision_latency_ms: float
    skill_attempts: Dict[str, int] = field(default_factory=dict)
    skill_successes: Dict[str, int] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)


class GrammarExecutor:
    def __init__(self, program: TaskProgram, env: Env, cfg: Optional[ExecutorConfig] = None,
                 risk_fn: Optional[Callable[[Atom, object], float]] = None):
        self.prog, self.env = program, env
        self.cfg = cfg or ExecutorConfig()
        self.risk_fn = risk_fn

    # ------------------------------------------------------------------ helpers
    def _geometric_producers(self, atom: Atom) -> Set[str]:
        out = set()
        for p in atom.pre:
            if p.name in GRIPPER_STATE:
                continue
            pid = self.prog.producer(atom.id, p)
            if pid not in (None, INITIAL_ID):
                out.add(pid)
        return out

    # ------------------------------------------------------------------ main loop
    def run(self) -> EpisodeResult:
        prog, env, cfg = self.prog, self.env, self.cfg
        completed: Set[str] = set()
        retries: Counter = Counter()
        attempts: Counter = Counter()
        successes: Counter = Counter()
        events: List[dict] = []
        n = dict(steps=0, local=0, full=0, stale=0, coll=0, guard=0)
        decision_s: List[float] = []
        termination = "max_steps"

        def log(kind: str, atom: Optional[str] = None, **kw):
            events.append({"step": n["steps"], "event": kind, "atom": atom, **kw})

        def local(ids: Set[str], reason: str, trigger: Optional[str] = None) -> None:
            redo = closure(prog.graph, ids) & completed
            completed.difference_update(redo)
            n["local"] += 1
            log("local_repair", trigger, reason=reason, redo=sorted(redo, key=prog.index.get))

        def full(reason: str) -> bool:
            n["full"] += 1
            completed.clear()
            retries.clear()
            log("full_replan", reason=reason)
            return n["full"] <= cfg.max_full_replans

        def execute(atom: Atom):
            res = env.execute(atom)
            n["steps"] += 1
            n["coll"] += res.collisions
            n["guard"] += res.guard_violations
            attempts[atom.skill.value] += 1
            return res

        for _ in range(cfg.max_steps * 5):  # guard against pathological repair loops
            if n["steps"] >= cfg.max_steps:
                break
            pending = [a.id for a in prog.atoms if a.id not in completed]
            if not pending:
                termination = "program_complete"
                break
            atom = prog.atom(pending[0])

            t0 = time.perf_counter()
            obs = env.observe()
            stale = stale_atoms(prog.atoms, completed, obs)
            missing = [] if stale else atom.failed_pre(obs)
            risk = self.risk_fn(atom, obs) if (self.risk_fn and not stale and not missing) else 0.0
            decision_s.append(time.perf_counter() - t0)

            # --- 2. scene changed under us
            if stale:
                n["stale"] += 1
                log("stale_effect", atoms=sorted(stale, key=prog.index.get))
                if cfg.local_repair:
                    local(stale, "stale_effect")
                elif not full("stale_effect"):
                    termination = "replan_budget"
                    break
                continue

            # --- 3. precondition not satisfied
            if missing:
                log("precondition_failed", atom.id, missing=[str(p) for p in missing])
                producers = {prog.producer(atom.id, p) for p in missing}
                unrecoverable = [p for p in missing if prog.producer(atom.id, p) in (None, INITIAL_ID)]
                if unrecoverable:
                    if all(p.name == "free" for p in unrecoverable) and obs.held is not None:
                        res = execute(recover_atom(obs.held))
                        log("recover", obs.held, info=res.info)
                        continue
                    if not full("unrecoverable_precondition"):
                        termination = "replan_budget"
                        break
                    continue
                if cfg.local_repair:
                    local(producers, "missing_precondition", atom.id)
                elif not full("missing_precondition"):
                    termination = "replan_budget"
                    break
                continue

            # --- WP4 hook: predicted violation before contact
            if risk > cfg.risk_threshold:
                log("predicted_violation", atom.id, risk=round(float(risk), 3))
                geo = self._geometric_producers(atom)
                if geo:
                    local(geo, "predicted_violation", atom.id)
                    continue

            # --- 4. execute
            res = execute(atom)
            obs = env.observe()
            failed = atom.failed_effect(obs)
            if res.ok and not failed:
                completed.add(atom.id)
                successes[atom.skill.value] += 1
                log("done", atom.id, info=res.info)
                continue

            # --- 5. effect failed
            retries[atom.id] += 1
            log("effect_failed", atom.id, failed=[str(p) for p in failed], info=res.info, ok=res.ok)
            if not cfg.local_repair or retries[atom.id] > cfg.max_atom_retries:
                if not full(f"{atom.id} failed {retries[atom.id]}x" if cfg.local_repair else "effect_failed"):
                    termination = "replan_budget"
                    break
                continue
            if retries[atom.id] >= 2 or not res.ok:
                still_valid = not atom.failed_pre(obs)
                geo = self._geometric_producers(atom)
                if still_valid and geo:
                    local(geo, "backoff_after_repeated_failure", atom.id)
            # otherwise: simply retry the same atom on the next iteration

        truth = env.true_state()
        success = all(g.holds(truth) for g in prog.goal)
        g0 = prog.goal[0]
        err = truth[g0.args[0]].xy_dist(truth[g0.args[1]]) * 1e3 if len(g0.args) == 2 else float("nan")
        if termination == "program_complete" and not success:
            termination = "complete_but_goal_false"

        return EpisodeResult(
            success=success,
            termination=termination,
            steps=n["steps"],
            local_repairs=n["local"],
            full_replans=n["full"],
            stale_detections=n["stale"],
            collisions=n["coll"],
            guard_violations=n["guard"],
            goal_xy_err_mm=float(err),
            decision_latency_ms=float(np.mean(decision_s) * 1e3) if decision_s else 0.0,
            skill_attempts=dict(attempts),
            skill_successes=dict(successes),
            events=events,
        )
