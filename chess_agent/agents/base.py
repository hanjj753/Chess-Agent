from abc import ABC, abstractmethod

import chess


class Agent(ABC):
    """Common interface for all chess agents."""

    name = "agent"

    @abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move | None:
        """Return a legal move for the current board, or None if no move exists."""

    def reset(self, *, seed: int | None = None) -> None:
        """Reset per-game state and optionally seed stochastic behavior."""
