"""Robot-independent contract for executing the existing PGAG atoms."""

from abc import ABC, abstractmethod
from grammar.atoms import Atom
from grammar.predicates import WorldState


class RobotAdapter(ABC):
    @abstractmethod
    def reset(self, task: str) -> None:
        """Create or reset the selected task scene."""

    @abstractmethod
    def get_world_state(self) -> WorldState:
        """Get the latest measured (or mock) PGAG world state."""

    @abstractmethod
    def execute_atom(self, atom: Atom) -> None:
        """Execute one PGAG action. Must fail loudly if unsupported."""

    def inject_displacement(self, obj: str) -> None:
        raise NotImplementedError(
            "This adapter does not support artificial displacement."
        )
