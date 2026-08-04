import chess

from chess_agent.engine.evaluation import PIECE_VALUES


def ordered_moves(
    board: chess.Board,
    preferred_move: chess.Move | None = None,
) -> list[chess.Move]:
    """Return legal moves sorted so alpha-beta can prune earlier."""
    return sorted(
        board.legal_moves,
        key=lambda move: move_order_score(board, move, preferred_move),
        reverse=True,
    )


def ordered_tactical_moves(board: chess.Board) -> list[chess.Move]:
    """Return noisy legal moves worth extending in quiescence search."""
    return sorted(
        (
            move
            for move in board.legal_moves
            if board.is_capture(move) or move.promotion is not None
        ),
        key=lambda move: move_order_score(board, move),
        reverse=True,
    )


def move_order_score(
    board: chess.Board,
    move: chess.Move,
    preferred_move: chess.Move | None = None,
) -> int:
    """Score moves only for search ordering, not for final evaluation."""
    score = 0

    if move == preferred_move:
        score += 1_000_000

    if board.is_capture(move):
        victim = captured_piece_type(board, move)
        attacker = board.piece_type_at(move.from_square)
        if victim is not None and attacker is not None:
            score += 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker]

    if move.promotion is not None:
        score += PIECE_VALUES[move.promotion]

    board.push(move)
    try:
        if board.is_check():
            score += 50
    finally:
        board.pop()

    return score


def captured_piece_type(board: chess.Board, move: chess.Move) -> chess.PieceType | None:
    if board.is_en_passant(move):
        return chess.PAWN

    piece = board.piece_at(move.to_square)
    if piece is None:
        return None
    return piece.piece_type
