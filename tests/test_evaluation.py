import chess

from chess_agent.engine.evaluation import (
    CHECKMATE_SCORE,
    evaluate,
    material_score,
    piece_square_score,
)


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


def test_knight_gets_bonus_for_better_white_square() -> None:
    edge_knight = chess.Board("4k3/8/8/8/8/7N/8/4K3 w - - 0 1")
    central_knight = chess.Board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")

    assert piece_square_score(central_knight, chess.WHITE) > piece_square_score(
        edge_knight,
        chess.WHITE,
    )


def test_knight_table_is_mirrored_for_black() -> None:
    edge_knight = chess.Board("4k3/8/7n/8/8/8/8/4K3 b - - 0 1")
    central_knight = chess.Board("4k3/8/5n2/8/8/8/8/4K3 b - - 0 1")

    assert piece_square_score(central_knight, chess.BLACK) > piece_square_score(
        edge_knight,
        chess.BLACK,
    )


def test_pawn_gets_bonus_for_advancing_toward_center() -> None:
    starting_pawn = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    advanced_pawn = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")

    assert piece_square_score(advanced_pawn, chess.WHITE) > piece_square_score(
        starting_pawn,
        chess.WHITE,
    )


def test_other_piece_tables_reward_active_squares() -> None:
    passive = chess.Board("6k1/8/8/8/8/8/B7/R5KQ w - - 0 1")
    active = chess.Board("6k1/4R3/8/8/2BQ4/8/8/6K1 w - - 0 1")

    assert piece_square_score(active, chess.WHITE) > piece_square_score(
        passive,
        chess.WHITE,
    )


def test_king_table_prefers_safer_early_square() -> None:
    central_king = chess.Board("6k1/8/8/8/4K3/8/8/8 w - - 0 1")
    safer_king = chess.Board("6k1/8/8/8/8/8/8/6K1 w - - 0 1")

    assert piece_square_score(safer_king, chess.WHITE) > piece_square_score(
        central_king,
        chess.WHITE,
    )
