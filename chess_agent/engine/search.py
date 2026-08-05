from dataclasses import dataclass
from time import perf_counter
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
    elapsed_seconds: float = 0.0
    timed_out: bool = False


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


class SearchTimeout(Exception):
    """Raised internally when a time-limited search runs out of time."""


@dataclass(frozen=True)
class SearchDeadline:
    expires_at: float | None

    @classmethod
    def from_time_limit(cls, time_limit: float | None) -> "SearchDeadline":
        if time_limit is None:
            return cls(expires_at=None)
        return cls(expires_at=perf_counter() + max(0.0, time_limit))

    def check(self) -> None:
        if self.expires_at is not None and perf_counter() >= self.expires_at:
            raise SearchTimeout


def find_best_move(
    board: chess.Board,
    depth: int,
    table: TranspositionTable | None = None,
    time_limit: float | None = None,
) -> SearchResult:
    """Search one move from the current board."""
    if depth < 1:
        raise ValueError("depth must be at least 1")

    started_at = perf_counter()
    deadline = SearchDeadline.from_time_limit(time_limit)
    if table is None:
        table = {}

    if time_limit is not None:
        return iterative_deepening_search(
            board=board,
            max_depth=depth,
            table=table,
            deadline=deadline,
            started_at=started_at,
        )

    stats = SearchStats()
    best_move, best_score = search_root(
        board=board,
        depth=depth,
        table=table,
        stats=stats,
        deadline=deadline,
    )
    return SearchResult(
        move=best_move,
        score=best_score,
        depth=depth,
        nodes=stats.nodes,
        table_hits=stats.table_hits,
        elapsed_seconds=perf_counter() - started_at,
        timed_out=False,
    )


def iterative_deepening_search(
    *,
    board: chess.Board,
    max_depth: int,
    table: TranspositionTable,
    deadline: SearchDeadline,
    started_at: float,
) -> SearchResult:
    aggregate_stats = SearchStats()
    last_result: SearchResult | None = None

    for current_depth in range(1, max_depth + 1):
        stats = SearchStats()
        try:
            best_move, best_score = search_root(
                board=board,
                depth=current_depth,
                table=table,
                stats=stats,
                deadline=deadline,
            )
        except SearchTimeout:
            aggregate_stats.nodes += stats.nodes
            aggregate_stats.table_hits += stats.table_hits
            return timed_search_result(
                board=board,
                previous=last_result,
                stats=aggregate_stats,
                started_at=started_at,
            )

        aggregate_stats.nodes += stats.nodes
        aggregate_stats.table_hits += stats.table_hits
        last_result = SearchResult(
            move=best_move,
            score=best_score,
            depth=current_depth,
            nodes=aggregate_stats.nodes,
            table_hits=aggregate_stats.table_hits,
            elapsed_seconds=perf_counter() - started_at,
            timed_out=False,
        )

    if last_result is None:
        return timed_search_result(
            board=board,
            previous=None,
            stats=aggregate_stats,
            started_at=started_at,
        )

    return last_result


def timed_search_result(
    *,
    board: chess.Board,
    previous: SearchResult | None,
    stats: SearchStats,
    started_at: float,
) -> SearchResult:
    if previous is not None:
        return SearchResult(
            move=previous.move,
            score=previous.score,
            depth=previous.depth,
            nodes=stats.nodes,
            table_hits=stats.table_hits,
            elapsed_seconds=perf_counter() - started_at,
            timed_out=True,
        )

    legal_moves = list(ordered_moves(board))
    move = legal_moves[0] if legal_moves else None
    return SearchResult(
        move=move,
        score=evaluate(board),
        depth=0,
        nodes=stats.nodes,
        table_hits=stats.table_hits,
        elapsed_seconds=perf_counter() - started_at,
        timed_out=True,
    )


def search_root(
    *,
    board: chess.Board,
    depth: int,
    table: TranspositionTable,
    stats: SearchStats,
    deadline: SearchDeadline,
) -> tuple[chess.Move | None, int]:
    best_move = None
    best_score = NEG_INF
    alpha = NEG_INF
    beta = POS_INF

    for move in ordered_moves(board):
        deadline.check()
        board.push(move)
        try:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                stats,
                table,
                deadline,
            )
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

    if best_move is None:
        best_score = evaluate(board)

    return best_move, best_score


def negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    stats: SearchStats,
    table: TranspositionTable | None = None,
    deadline: SearchDeadline | None = None,
) -> int:
    """Return the position score from the side-to-move perspective."""
    if table is None:
        table = {}
    if deadline is None:
        deadline = SearchDeadline(expires_at=None)

    deadline.check()
    stats.nodes += 1

    if board.is_game_over(claim_draw=True):
        return evaluate(board)

    if depth == 0:
        return quiescence(board, alpha, beta, stats, deadline)

    original_alpha = alpha
    original_beta = beta
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
        try:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                stats,
                table,
                deadline,
            )
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

        if alpha >= beta:
            break

    if best_score <= original_alpha:
        bound: BoundType = "upper"
    elif best_score >= original_beta:
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
    deadline: SearchDeadline | None = None,
) -> int:
    """Search only tactical continuations before evaluating a quiet position."""
    if deadline is None:
        deadline = SearchDeadline(expires_at=None)

    deadline.check()
    stats.nodes += 1

    if board.is_game_over(claim_draw=True):
        return evaluate(board)

    if board.is_check():
        best_score = NEG_INF
        for move in ordered_moves(board):
            deadline.check()
            board.push(move)
            try:
                score = -quiescence(board, -beta, -alpha, stats, deadline)
            finally:
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
        deadline.check()
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, stats, deadline)
        finally:
            board.pop()

        if score >= beta:
            return score

        alpha = max(alpha, score)

    return alpha
