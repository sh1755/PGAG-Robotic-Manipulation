"""Dependency graph over atoms and dependency-closure utilities.

An edge i -> j means atom j has a precondition that atom i established (the
most recent producer of that predicate before j, with no deletion in between).
Local repair invalidates the *closure* of a failed atom: the atom itself plus
all of its descendants, i.e. the minimal subgraph whose assumptions are broken.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set

import networkx as nx

from .atoms import Atom
from .predicates import Predicate, WorldState

INITIAL = -1  # predicate is supplied by the initial state, not by any atom


def find_producer(atoms: Sequence[Atom], j: int, pred: Predicate) -> Optional[int]:
    """Index of the atom that establishes ``pred`` for atom ``j``.

    Returns INITIAL if no earlier atom touches it, or None if the most recent
    earlier atom touching it *deletes* it (an invalid program).
    """
    for i in range(j - 1, -1, -1):
        if pred in atoms[i].effect:
            return i
        if pred in atoms[i].deletes:
            return None
    return INITIAL


def build_dependency_graph(atoms: Sequence[Atom]) -> nx.DiGraph:
    g = nx.DiGraph()
    for a in atoms:
        g.add_node(a.id, skill=a.skill.value, objects=a.objects)
    for j, aj in enumerate(atoms):
        for p in aj.pre:
            i = find_producer(atoms, j, p)
            if i is not None and i >= 0:
                src = atoms[i].id
                if g.has_edge(src, aj.id):
                    g.edges[src, aj.id]["predicates"].append(str(p))
                else:
                    g.add_edge(src, aj.id, predicates=[str(p)])
    return g


def closure(g: nx.DiGraph, ids: Iterable[str]) -> Set[str]:
    """ids plus every atom downstream of them."""
    out: Set[str] = set()
    for i in ids:
        out.add(i)
        out |= nx.descendants(g, i)
    return out


def stale_atoms(atoms: Sequence[Atom], completed: Set[str], s: WorldState) -> Set[str]:
    """Completed atoms whose persistent effects no longer hold in ``s``.

    An effect only counts if no later *completed* atom deliberately deleted it.
    """
    stale: Set[str] = set()
    for idx, a in enumerate(atoms):
        if a.id not in completed:
            continue
        for e in a.effect:
            if not e.persistent:
                continue
            deleted_later = any(
                atoms[k].id in completed and e in atoms[k].deletes for k in range(idx + 1, len(atoms))
            )
            if not deleted_later and not e.holds(s):
                stale.add(a.id)
                break
    return stale
