import math
import random

import chess

from chess_agent.agents.alphabeta_agent import AlphaBetaAgent
from chess_agent.agents.base import Agent


class AlphaRandomAgent(Agent):
    """Choose an alpha-beta move with a fixed probability, otherwise at random."""

    name = "alpha-random"

    def __init__(
        self,
        *,
        alpha_move_probability: float = 0.1,
        depth: int = 1,
        time_limit: float | None = None,
    ) -> None:
        if not math.isfinite(alpha_move_probability) or not (
            0.0 <= alpha_move_probability <= 1.0
        ):
            raise ValueError("alpha_move_probability must be between 0 and 1")
        self.alpha_move_probability = alpha_move_probability
        self.alpha_agent = AlphaBetaAgent(depth=depth, time_limit=time_limit)
        self._random = random.Random()

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._random.seed(seed)
        self.alpha_agent.clear_cache()

    def select_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if self._random.random() < self.alpha_move_probability:
            return self.alpha_agent.select_move(board)
        return self._random.choice(legal_moves)
