import chess

from chess_agent.engine.evaluation import (
    CHECKMATE_SCORE,
    bishop_pair_score,
    evaluate,
    phase_weights,
    king_development_score,
    king_safety_score,
    material_score,
    mobility_score,
    pawn_structure_score,
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
    central_king = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4K3/8/PPPPPPPP/RNBQ1BNR w kq - 0 1"
    )
    safer_king = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BKR w kq - 0 1"
    )

    assert piece_square_score(safer_king, chess.WHITE) > piece_square_score(
        central_king,
        chess.WHITE,
    )


def test_phase_weights_start_in_opening() -> None:
    phase = phase_weights(chess.Board())

    assert phase.opening == 1.0
    assert phase.middlegame == 0.0
    assert phase.endgame == 0.0


def test_phase_weights_detect_endgame_by_material() -> None:
    board = chess.Board("6k1/8/8/8/4K3/8/4P3/8 w - - 0 1")
    phase = phase_weights(board)

    assert phase.endgame > 0.9
    assert phase.opening == 0.0


def test_endgame_king_table_rewards_active_king() -> None:
    passive = chess.Board("6k1/8/8/8/8/8/8/6K1 w - - 0 1")
    active = chess.Board("6k1/8/8/8/4K3/8/8/8 w - - 0 1")

    assert piece_square_score(active, chess.WHITE) > piece_square_score(
        passive,
        chess.WHITE,
    )


def test_opening_king_development_penalizes_early_king_walk() -> None:
    safe = chess.Board()
    exposed = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPKPPP/RNBQ1BNR w kq - 0 1"
    )

    assert king_development_score(safe, chess.WHITE) > king_development_score(
        exposed,
        chess.WHITE,
    )


def test_opening_king_development_rewards_castling() -> None:
    uncastled = chess.Board()
    castled = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 w kq - 0 1"
    )

    assert king_development_score(castled, chess.WHITE) > king_development_score(
        uncastled,
        chess.WHITE,
    )


def test_mobility_rewards_more_legal_moves() -> None:
    open_rook = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    blocked_rook = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1")

    assert mobility_score(open_rook, chess.WHITE) > mobility_score(
        blocked_rook,
        chess.WHITE,
    )


def test_bishop_pair_gets_bonus() -> None:
    one_bishop = chess.Board("4k3/8/8/8/8/8/8/B3K3 w - - 0 1")
    two_bishops = chess.Board("4k3/8/8/8/8/8/8/BB2K3 w - - 0 1")

    assert bishop_pair_score(two_bishops, chess.WHITE) > bishop_pair_score(
        one_bishop,
        chess.WHITE,
    )


def test_pawn_structure_penalizes_doubled_pawns() -> None:
    healthy = chess.Board("4k3/8/8/8/8/8/P1P5/4K3 w - - 0 1")
    doubled = chess.Board("4k3/8/8/8/8/P7/P7/4K3 w - - 0 1")

    assert pawn_structure_score(healthy, chess.WHITE) > pawn_structure_score(
        doubled,
        chess.WHITE,
    )


def test_pawn_structure_rewards_passed_pawn() -> None:
    blocked = chess.Board("4k3/8/4p3/8/4P3/8/8/4K3 w - - 0 1")
    passed = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")

    assert pawn_structure_score(passed, chess.WHITE) > pawn_structure_score(
        blocked,
        chess.WHITE,
    )


def test_king_safety_rewards_pawn_shield() -> None:
    exposed = chess.Board("6k1/8/8/8/8/8/8/6K1 w - - 0 1")
    shielded = chess.Board("6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1")

    assert king_safety_score(shielded, chess.WHITE) > king_safety_score(
        exposed,
        chess.WHITE,
    )
