"""Minimal task-specific repair selector for an early G1 adapter prototype.

For this scaffold, only recovering the relation on(b,a) during the stack
program is supported. It is NOT a full graph-based PGAG recovery engine.
"""


class G1LocalRecovery:
    def repair_on_b_a(self, program, run_checked):
        chain = [
            a for a in program.atoms
            if a.id.startswith("b_")
        ]
        if len(chain) != 6:
            raise RuntimeError(
                "Unexpected stack grammar: expected 6 Block 2 actions"
            )
        for atom in chain:
            run_checked(atom)
        return [atom.id for atom in chain]
