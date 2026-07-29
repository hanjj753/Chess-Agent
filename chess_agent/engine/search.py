from dataclasses import dataclass

import chess

from chess_agent.engine.evaluation import evaluate
from chess_agent.engine.move_ordering import ordered_moves

NEG_INF = -10**9
POS_INF = 10**9


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    nodes: int


@dataclass
class SearchStats:
    nodes: int = 0


def find_best_move(board: chess.Board, depth: int) -> SearchResult:
    """Search one move from the current board."""
    if depth < 1:
        raise ValueError("depth must be at least 1")

    best_move = None
    best_score = NEG_INF
    alpha = NEG_INF
    beta = POS_INF
    stats = SearchStats()

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, stats)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

    if best_move is None:
        best_score = evaluate(board)

    return SearchResult(
        move=best_move,
        score=best_score,
        depth=depth,
        nodes=stats.nodes,
    )


def negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    stats: SearchStats,
) -> int:
    """Return the position score from the side-to-move perspective."""
    stats.nodes += 1

    if depth == 0 or board.is_game_over(claim_draw=True):
        return evaluate(board)

    best_score = NEG_INF

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, stats)
        board.pop()

        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best_score
