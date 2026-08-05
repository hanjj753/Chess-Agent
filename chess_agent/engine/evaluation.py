import chess

CHECKMATE_SCORE = 100_000
MOBILITY_WEIGHT = 2
BISHOP_PAIR_BONUS = 30
DOUBLED_PAWN_PENALTY = 10
ISOLATED_PAWN_PENALTY = 10
PASSED_PAWN_BASE_BONUS = 10
KING_SHIELD_FRONT_BONUS = 8
KING_SHIELD_SECOND_BONUS = 4
KING_ATTACK_PENALTY = 3

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   5,   5,   0,   0,   0,
]

QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_TABLE = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


def evaluate(board: chess.Board) -> int:
    """Return a score from the side-to-move perspective."""
    if board.is_checkmate():
        return -CHECKMATE_SCORE

    if board.is_game_over(claim_draw=True):
        return 0

    perspective = board.turn
    return (
        material_score(board, perspective)
        + piece_square_score(board, perspective)
        + mobility_score(board, perspective)
        + king_safety_score(board, perspective)
        + pawn_structure_score(board, perspective)
        + bishop_pair_score(board, perspective)
    )


def material_score(board: chess.Board, perspective: chess.Color) -> int:
    """Return material balance from `perspective`."""
    white_score = 0
    black_score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_score += len(board.pieces(piece_type, chess.WHITE)) * value
        black_score += len(board.pieces(piece_type, chess.BLACK)) * value

    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def piece_square_score(board: chess.Board, perspective: chess.Color) -> int:
    """Return piece square score from `perspective`."""
    white_score = 0
    black_score = 0

    for piece_type, table in PIECE_SQUARE_TABLES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            white_score += table[square]

        for square in board.pieces(piece_type, chess.BLACK):
            black_score += table[chess.square_mirror(square)]

    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def mobility_score(board: chess.Board, perspective: chess.Color) -> int:
    """Reward positions where a side has more legal move choices."""
    white_mobility = legal_move_count(board, chess.WHITE)
    black_mobility = legal_move_count(board, chess.BLACK)
    score = (white_mobility - black_mobility) * MOBILITY_WEIGHT
    return score if perspective == chess.WHITE else -score


def legal_move_count(board: chess.Board, color: chess.Color) -> int:
    board_copy = board.copy(stack=False)
    board_copy.turn = color
    return board_copy.legal_moves.count()


def bishop_pair_score(board: chess.Board, perspective: chess.Color) -> int:
    white_score = BISHOP_PAIR_BONUS if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0
    black_score = BISHOP_PAIR_BONUS if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0
    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def pawn_structure_score(board: chess.Board, perspective: chess.Color) -> int:
    white_score = pawn_structure_for_color(board, chess.WHITE)
    black_score = pawn_structure_for_color(board, chess.BLACK)
    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def pawn_structure_for_color(board: chess.Board, color: chess.Color) -> int:
    score = 0
    pawns_by_file = [
        list(board.pieces(chess.PAWN, color) & chess.BB_FILES[file_index])
        for file_index in range(8)
    ]

    for pawns in pawns_by_file:
        if len(pawns) > 1:
            score -= DOUBLED_PAWN_PENALTY * (len(pawns) - 1)

    for square in board.pieces(chess.PAWN, color):
        file_index = chess.square_file(square)
        adjacent_files = [
            adjacent_file
            for adjacent_file in (file_index - 1, file_index + 1)
            if 0 <= adjacent_file < 8
        ]

        if all(not pawns_by_file[adjacent_file] for adjacent_file in adjacent_files):
            score -= ISOLATED_PAWN_PENALTY

        if is_passed_pawn(board, square, color):
            score += PASSED_PAWN_BASE_BONUS + passed_pawn_progress(square, color) * 2

    return score


def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    enemy = not color
    candidate_files = [
        candidate_file
        for candidate_file in (file_index - 1, file_index, file_index + 1)
        if 0 <= candidate_file < 8
    ]

    if color == chess.WHITE:
        candidate_ranks = range(rank_index + 1, 8)
    else:
        candidate_ranks = range(rank_index - 1, -1, -1)

    for candidate_file in candidate_files:
        for candidate_rank in candidate_ranks:
            piece = board.piece_at(chess.square(candidate_file, candidate_rank))
            if piece == chess.Piece(chess.PAWN, enemy):
                return False
    return True


def passed_pawn_progress(square: chess.Square, color: chess.Color) -> int:
    rank_index = chess.square_rank(square)
    return rank_index if color == chess.WHITE else 7 - rank_index


def king_safety_score(board: chess.Board, perspective: chess.Color) -> int:
    white_score = king_safety_for_color(board, chess.WHITE)
    black_score = king_safety_for_color(board, chess.BLACK)
    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def king_safety_for_color(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0

    score = pawn_shield_score(board, king_square, color)
    enemy = not color
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        score -= len(board.attackers(enemy, square)) * KING_ATTACK_PENALTY
    return score


def pawn_shield_score(
    board: chess.Board,
    king_square: chess.Square,
    color: chess.Color,
) -> int:
    score = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    direction = 1 if color == chess.WHITE else -1

    for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        first_rank = king_rank + direction
        second_rank = king_rank + 2 * direction

        if 0 <= first_rank < 8:
            piece = board.piece_at(chess.square(file_index, first_rank))
            if piece == chess.Piece(chess.PAWN, color):
                score += KING_SHIELD_FRONT_BONUS

        if 0 <= second_rank < 8:
            piece = board.piece_at(chess.square(file_index, second_rank))
            if piece == chess.Piece(chess.PAWN, color):
                score += KING_SHIELD_SECOND_BONUS

    return score


def explain_evaluation_targets() -> list[str]:
    """Tiny study guide for the next features to implement."""
    return [
        "piece-square tables",
        "mobility",
        "king safety",
        "pawn structure",
        "bishop pair",
    ]
