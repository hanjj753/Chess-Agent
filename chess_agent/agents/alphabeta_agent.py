import chess

from chess_agent.agents.base import Agent
from chess_agent.engine.search import SearchResult, find_best_move


class AlphaBetaAgent(Agent):
    name = "alpha"

    def __init__(self, depth: int = 3):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.last_result: SearchResult | None = None

    def select_move(self, board: chess.Board) -> chess.Move | None:
        self.last_result = find_best_move(board, depth=self.depth)
        return self.last_result.move
