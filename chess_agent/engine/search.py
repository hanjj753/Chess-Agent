from dataclasses import dataclass
from typing import Hashable, Literal

import chess

from chess_agent.engine.evaluation import evaluate
from chess_agent.engine.move_ordering import ordered_moves, ordered_tactical_moves

NEG_INF = -10**9
POS_INF = 10**9


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    nodes: int
    table_hits: int


@dataclass
class SearchStats:
    nodes: int = 0
    table_hits: int = 0


BoundType = Literal["exact", "lower", "upper"]


@dataclass(frozen=True)
class TranspositionEntry:
    depth: int
    score: int
    bound: BoundType
    best_move: chess.Move | None = None


TranspositionTable = dict[Hashable, TranspositionEntry]


def find_best_move(
    board: chess.Board,
    depth: int,
    table: TranspositionTable | None = None,
) -> SearchResult:
    """Search one move from the current board."""
    if depth < 1:
        raise ValueError("depth must be at least 1")

    best_move = None
    best_score = NEG_INF
    alpha = NEG_INF
    beta = POS_INF
    stats = SearchStats()
    if table is None:
        table = {}

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, stats, table)
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
        table_hits=stats.table_hits,
    )


def negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    stats: SearchStats,
    table: TranspositionTable | None = None,
) -> int:
    """Return the position score from the side-to-move perspective."""
    if table is None:
        table = {}

    stats.nodes += 1

    if board.is_game_over(claim_draw=True):
        return evaluate(board)

    if depth == 0:
        return quiescence(board, alpha, beta, stats)

    original_alpha = alpha
    key = board_key(board)
    cached_best_move = None
    cached_entry = table.get(key)
    if cached_entry is not None and cached_entry.depth >= depth:
        stats.table_hits += 1
        cached_best_move = cached_entry.best_move

        if cached_entry.bound == "exact":
            return cached_entry.score
        if cached_entry.bound == "lower":
            alpha = max(alpha, cached_entry.score)
        elif cached_entry.bound == "upper":
            beta = min(beta, cached_entry.score)

        if alpha >= beta:
            return cached_entry.score

    best_score = NEG_INF
    best_move = None

    for move in ordered_moves(board, preferred_move=cached_best_move):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, stats, table)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

        if alpha >= beta:
            break

    if best_score <= original_alpha:
        bound: BoundType = "upper"
    elif best_score >= beta:
        bound = "lower"
    else:
        bound = "exact"

    table[key] = TranspositionEntry(
        depth=depth,
        score=best_score,
        bound=bound,
        best_move=best_move,
    )
    return best_score


def board_key(board: chess.Board) -> Hashable:
    """Return a hashable key for positions that can share search results."""
    key = board._transposition_key()
    return key, board.halfmove_clock


def quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    stats: SearchStats,
) -> int:
    """Search only tactical continuations before evaluating a quiet position."""
    stats.nodes += 1

    if board.is_game_over(claim_draw=True):
        return evaluate(board)

    if board.is_check():
        best_score = NEG_INF
        for move in ordered_moves(board):
            board.push(move)
            score = -quiescence(board, -beta, -alpha, stats)
            board.pop()

            if score >= beta:
                return score

            best_score = max(best_score, score)
            alpha = max(alpha, score)

        return best_score

    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return stand_pat

    alpha = max(alpha, stand_pat)

    for move in ordered_tactical_moves(board):
        board.push(move)
        score = -quiescence(board, -beta, -alpha, stats)
        board.pop()

        if score >= beta:
            return score

        alpha = max(alpha, score)

    return alpha
