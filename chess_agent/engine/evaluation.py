import chess

CHECKMATE_SCORE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate(board: chess.Board) -> int:
    """Return a score from the side-to-move perspective."""
    if board.is_checkmate():
        return -CHECKMATE_SCORE

    if board.is_game_over(claim_draw=True):
        return 0

    return material_score(board, board.turn)


def material_score(board: chess.Board, perspective: chess.Color) -> int:
    """Return material balance from `perspective`."""
    white_score = 0
    black_score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_score += len(board.pieces(piece_type, chess.WHITE)) * value
        black_score += len(board.pieces(piece_type, chess.BLACK)) * value

    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def explain_evaluation_targets() -> list[str]:
    """Tiny study guide for the next features to implement."""
    return [
        "piece-square tables",
        "mobility",
        "king safety",
        "pawn structure",
        "bishop pair",
    ]
