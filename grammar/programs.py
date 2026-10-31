"""Task programs (hand-written for Week 1) for the three agreed tasks.

Each program is an ordered list of atoms plus goal predicates. The dependency
graph is derived automatically from pre/effect/delete lists, so if you edit a
program, run ``python -m grammar.show --task <name>`` to see the new graph and
``pytest`` to check that it is still symbolically valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .atoms import Atom, Guards, Skill
from .dependency import INITIAL, build_dependency_graph, find_producer
from .predicates import P, Predicate

INITIAL_ID = "<initial>"


@dataclass
class TaskProgram:
    name: str
    atoms: List[Atom]
    goal: List[Predicate]
    initial: List[Predicate] = field(default_factory=list)

    def __post_init__(self):
        ids = [a.id for a in self.atoms]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate atom ids in program '{self.name}'")
        self.index: Dict[str, int] = {a.id: i for i, a in enumerate(self.atoms)}
        self.graph = build_dependency_graph(self.atoms)

    def atom(self, aid: str) -> Atom:
        return self.atoms[self.index[aid]]

    def producer(self, aid: str, pred: Predicate) -> Optional[str]:
        """Atom id that establishes ``pred`` for atom ``aid``; INITIAL_ID or None otherwise."""
        i = find_producer(self.atoms, self.index[aid], pred)
        if i is None:
            return None
        if i == INITIAL:
            return INITIAL_ID if pred in self.initial else None
        return self.atoms[i].id

    def validate(self) -> List[str]:
        """Symbolic forward simulation (STRIPS). Returns a list of problems; empty = valid."""
        problems: List[str] = []
        facts = set(self.initial)
        for a in self.atoms:
            for p in a.pre:
                if p not in facts:
                    problems.append(f"{a.id}: precondition {p} not established")
            facts -= set(a.deletes)
            facts |= set(a.effect)
        for g in self.goal:
            if g not in facts:
                problems.append(f"goal {g} not established by the program")
        return problems

    def summary(self) -> str:
        lines = [f"Task program: {self.name}", "-" * 60]
        for a in self.atoms:
            lines.append(str(a))
            lines.append(f"    pre:     {', '.join(map(str, a.pre)) or '-'}")
            lines.append(f"    effect:  {', '.join(map(str, a.effect)) or '-'}")
            if a.deletes:
                lines.append(f"    deletes: {', '.join(map(str, a.deletes))}")
        lines.append(f"goal: {', '.join(map(str, self.goal))}")
        lines.append("-" * 60 + "\nDependency edges (producer -> consumer : predicates)")
        for u, v, d in self.graph.edges(data=True):
            lines.append(f"    {u} -> {v} : {', '.join(d['predicates'])}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        out = [f'digraph "{self.name}" {{', "  rankdir=LR; node [shape=box, fontname=Helvetica];"]
        for a in self.atoms:
            out.append(f'  "{a.id}" [label="{a.skill.value}\\n{",".join(a.objects)}"];')
        for u, v, d in self.graph.edges(data=True):
            out.append(f'  "{u}" -> "{v}" [label="{"; ".join(d["predicates"])}", fontsize=9];')
        out.append("}")
        return "\n".join(out)


# --------------------------------------------------------------------------- building blocks
def _pick(obj: str, prefix: str, lift_h: float = 0.12) -> List[Atom]:
    free, near, hold = P("free"), P("near_ee", obj), P("holding", obj)
    lifted = P("lifted", obj, h=0.08)
    return [
        Atom(f"{prefix}approach", Skill.APPROACH, (obj,), pre=[free], effect=[near],
             omega=Guards(min_clearance=0.01)),
        Atom(f"{prefix}grasp", Skill.GRASP, (obj,), pre=[free, near], effect=[hold], deletes=[free],
             omega=Guards(max_force=40.0)),
        Atom(f"{prefix}lift", Skill.LIFT, (obj,), pre=[hold], effect=[lifted],
             params={"height": lift_h}),
    ]


# --------------------------------------------------------------------------- task 1: pick-place
def pick_place(obj: str = "cube", bin_: str = "bin") -> TaskProgram:
    hold, lifted = P("holding", obj), P("lifted", obj, h=0.08)
    above, inside = P("above", obj, bin_), P("inside", obj, bin_)
    atoms = _pick(obj, "") + [
        Atom("transport", Skill.TRANSPORT, (obj, bin_), pre=[hold, lifted], effect=[above]),
        Atom("release", Skill.RELEASE, (obj, bin_), pre=[hold, above], effect=[inside, P("free")],
             deletes=[hold]),
    ]
    return TaskProgram("pick_place", atoms, goal=[inside], initial=[P("free")])


# --------------------------------------------------------------------------- task 2: stacking
def _place_on(obj: str, target: str, prefix: str, extra_pre: Optional[List[Predicate]] = None) -> List[Atom]:
    hold, lifted = P("holding", obj), P("lifted", obj, h=0.08)
    above = P("above", obj, target)
    aligned = P("aligned", obj, target, tol_xy=0.005, tol_deg=5.0)
    on = P("on", obj, target)
    return _pick(obj, prefix) + [
        Atom(f"{prefix}transport", Skill.TRANSPORT, (obj, target), pre=[hold, lifted] + (extra_pre or []),
             effect=[above]),
        Atom(f"{prefix}align", Skill.ALIGN, (obj, target), pre=[hold, above], effect=[aligned],
             omega=Guards(pos_tol=0.005, rot_tol_deg=5.0)),
        Atom(f"{prefix}release", Skill.RELEASE, (obj, target), pre=[hold, aligned], effect=[on, P("free")],
             deletes=[hold]),
    ]


def stack(a: str = "a", b: str = "b", c: str = "c") -> TaskProgram:
    """Place b on a, then c on b. c's transport requires on(b, a) still to hold."""
    atoms = _place_on(b, a, "b_") + _place_on(c, b, "c_", extra_pre=[P("on", b, a)])
    return TaskProgram("stack", atoms, goal=[P("on", b, a), P("on", c, b)], initial=[P("free")])


# --------------------------------------------------------------------------- task 3: peg-in-hole
def peg_insert(peg: str = "peg", hole: str = "hole", depth: float = 0.015) -> TaskProgram:
    hold, lifted = P("holding", peg), P("lifted", peg, h=0.08)
    above = P("above", peg, hole)
    aligned = P("aligned", peg, hole, tol_xy=0.003, tol_deg=3.0)
    inserted = P("inserted", peg, hole, depth=depth)
    atoms = _pick(peg, "") + [
        Atom("transport", Skill.TRANSPORT, (peg, hole), pre=[hold, lifted], effect=[above]),
        Atom("align", Skill.ALIGN, (peg, hole), pre=[hold, above], effect=[aligned],
             omega=Guards(pos_tol=0.002, rot_tol_deg=3.0)),
        Atom("insert", Skill.INSERT, (peg, hole), pre=[hold, aligned], effect=[inserted],
             omega=Guards(max_force=15.0, pos_tol=0.002), params={"depth": 0.02}),
        Atom("release", Skill.RELEASE, (peg, hole), pre=[hold, inserted], effect=[P("free")],
             deletes=[hold]),
    ]
    return TaskProgram("peg_insert", atoms, goal=[inserted], initial=[P("free")])


TASKS: Dict[str, Callable[[], TaskProgram]] = {
    "pick_place": pick_place,
    "stack": stack,
    "peg_insert": peg_insert,
}
