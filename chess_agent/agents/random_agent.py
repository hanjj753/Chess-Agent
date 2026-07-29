import random

import chess

from chess_agent.agents.base import Agent


class RandomAgent(Agent):
    name = "random"

    def select_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)
