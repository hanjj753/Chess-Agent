import chess

from chess_agent.agents.base import Agent
from chess_agent.engine.search import SearchResult, TranspositionTable, find_best_move


class AlphaBetaAgent(Agent):
    name = "alpha"

    def __init__(self, depth: int = 3, time_limit: float | None = None):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.time_limit = time_limit
        self.last_result: SearchResult | None = None
        self.transposition_table: TranspositionTable = {}

    def select_move(self, board: chess.Board) -> chess.Move | None:
        self.last_result = find_best_move(
            board,
            depth=self.depth,
            table=self.transposition_table,
            time_limit=self.time_limit,
        )
        return self.last_result.move

    def clear_cache(self) -> None:
        self.transposition_table.clear()
