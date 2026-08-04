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


def test_alpha_beta_agent_reuses_cache_between_searches() -> None:
    board = chess.Board()
    agent = AlphaBetaAgent(depth=2)

    agent.select_move(board)
    first_hits = agent.last_result.table_hits

    agent.select_move(board)

    assert agent.last_result.table_hits > first_hits
