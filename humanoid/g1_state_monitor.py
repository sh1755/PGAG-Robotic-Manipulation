"""Evaluate the existing PGAG predicate objects against an adapter state."""

from grammar.predicates import P, Predicate


class G1StateMonitor:
    def __init__(self, adapter):
        self.adapter = adapter

    def unmet(self, predicates):
        state = self.adapter.get_world_state()
        return [str(p) for p in predicates if not p.holds(state)]

    def on(self, top, bottom):
        return P("on", top, bottom).holds(self.adapter.get_world_state())

    def report(self, program):
        return {
            "goals_met": not self.unmet(program.goal),
            "missing_goals": self.unmet(program.goal),
        }
