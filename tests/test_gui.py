import chess

from chess_agent.agents import HumanAgent, RandomAgent
from chess_agent.gui import choose_orientation, display_square, move_from_squares, square_position


def test_display_square_white_orientation() -> None:
    assert display_square(0, 0, chess.WHITE) == chess.A8
    assert display_square(7, 7, chess.WHITE) == chess.H1


def test_display_square_black_orientation() -> None:
    assert display_square(0, 0, chess.BLACK) == chess.H1
    assert display_square(7, 7, chess.BLACK) == chess.A8


def test_square_position_round_trips_for_both_orientations() -> None:
    for orientation in (chess.WHITE, chess.BLACK):
        for square in (chess.A1, chess.E4, chess.H8):
            row, col = square_position(square, orientation)
            assert display_square(row, col, orientation) == square


def test_move_from_squares_returns_legal_move() -> None:
    board = chess.Board()

    assert move_from_squares(board, chess.E2, chess.E4) == chess.Move.from_uci("e2e4")


def test_move_from_squares_prefers_queen_promotion() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    assert move_from_squares(board, chess.A7, chess.A8) == chess.Move.from_uci("a7a8q")


def test_choose_orientation_follows_black_human() -> None:
    agents = {
        chess.WHITE: RandomAgent(),
        chess.BLACK: HumanAgent(input_fn=lambda _: "quit", output_fn=lambda _: None),
    }

    assert choose_orientation(agents) == chess.BLACK
