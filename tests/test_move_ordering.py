import chess

from chess_agent.engine.move_ordering import move_order_score, ordered_moves


def test_capture_scores_above_quiet_move() -> None:
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    capture = chess.Move.from_uci("e4d5")
    quiet = chess.Move.from_uci("e1f1")

    assert move_order_score(board, capture) > move_order_score(board, quiet)


def test_ordered_moves_keeps_all_legal_moves() -> None:
    board = chess.Board()

    assert set(ordered_moves(board)) == set(board.legal_moves)
