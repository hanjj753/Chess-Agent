import chess

from chess_agent.engine.evaluation import CHECKMATE_SCORE, evaluate, material_score


def test_initial_position_is_equal() -> None:
    board = chess.Board()

    assert material_score(board, chess.WHITE) == 0
    assert material_score(board, chess.BLACK) == 0
    assert evaluate(board) == 0


def test_material_score_uses_requested_perspective() -> None:
    board = chess.Board()
    board.remove_piece_at(chess.D8)

    assert material_score(board, chess.WHITE) == 900
    assert material_score(board, chess.BLACK) == -900


def test_checkmated_side_gets_negative_score() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert board.is_checkmate()
    assert evaluate(board) == -CHECKMATE_SCORE
