import chess

from chess_agent.engine.evaluation import evaluate
from chess_agent.agents import AlphaBetaAgent
from chess_agent.engine.search import (
    NEG_INF,
    POS_INF,
    SearchStats,
    find_best_move,
    negamax,
    quiescence,
)


def test_quiescence_finds_available_recapture() -> None:
    board = chess.Board("3Qk3/8/8/8/8/8/8/4K3 b - - 0 1")
    stats = SearchStats()

    assert quiescence(board, NEG_INF, POS_INF, stats) > evaluate(board)


def test_quiescence_keeps_quiet_position_static() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    stats = SearchStats()

    assert quiescence(board, NEG_INF, POS_INF, stats) == evaluate(board)


def test_quiescence_searches_check_evasions() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    stats = SearchStats()

    assert board.is_check()
    quiescence(board, NEG_INF, POS_INF, stats)
    assert stats.nodes > 1


def test_transposition_table_reuses_same_position_at_same_depth() -> None:
    board = chess.Board()
    stats = SearchStats()
    table = {}

    first_score = negamax(board, 1, NEG_INF, POS_INF, stats, table)
    first_hits = stats.table_hits

    second_score = negamax(board, 1, NEG_INF, POS_INF, stats, table)

    assert second_score == first_score
    assert stats.table_hits == first_hits + 1


def test_transposition_table_reuses_deeper_entry_for_shallower_search() -> None:
    board = chess.Board()
    stats = SearchStats()
    table = {}

    deeper_score = negamax(board, 2, NEG_INF, POS_INF, stats, table)
    first_hits = stats.table_hits

    shallower_score = negamax(board, 1, NEG_INF, POS_INF, stats, table)

    assert shallower_score == deeper_score
    assert stats.table_hits == first_hits + 1


def test_find_best_move_reports_table_hits() -> None:
    board = chess.Board()

    result = find_best_move(board, depth=3)

    assert result.nodes > 0
    assert result.table_hits >= 0


def test_time_limited_search_falls_back_to_legal_move() -> None:
    board = chess.Board()

    result = find_best_move(board, depth=8, time_limit=0.0)

    assert result.move in board.legal_moves
    assert result.depth == 0
    assert result.timed_out


def test_time_limited_search_does_not_mutate_board_on_timeout() -> None:
    board = chess.Board()
    original_fen = board.fen()

    find_best_move(board, depth=8, time_limit=0.0)

    assert board.fen() == original_fen


def test_time_limited_search_reports_completed_depth() -> None:
    board = chess.Board()

    result = find_best_move(board, depth=2, time_limit=1.0)

    assert result.move in board.legal_moves
    assert result.depth >= 1
    assert result.elapsed_seconds >= 0


def test_alpha_beta_agent_reuses_cache_between_searches() -> None:
    board = chess.Board()
    agent = AlphaBetaAgent(depth=2)

    agent.select_move(board)
    first_hits = agent.last_result.table_hits

    agent.select_move(board)

    assert agent.last_result.table_hits > first_hits


def test_alpha_beta_agent_accepts_time_limit() -> None:
    board = chess.Board()
    agent = AlphaBetaAgent(depth=8, time_limit=0.01)

    move = agent.select_move(board)

    assert move in board.legal_moves
    assert agent.last_result.elapsed_seconds >= 0
