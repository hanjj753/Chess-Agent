from dataclasses import dataclass

import chess

CHECKMATE_SCORE = 100_000
MAX_OPENING_PLY = 24
MAX_MATERIAL_PHASE = 24
MOBILITY_WEIGHT = 1
BISHOP_PAIR_BONUS = 20
DOUBLED_PAWN_PENALTY = 10
ISOLATED_PAWN_PENALTY = 10
PASSED_PAWN_BASE_BONUS = 8
KING_SHIELD_FRONT_BONUS = 4
KING_SHIELD_SECOND_BONUS = 2
KING_ATTACK_PENALTY = 2
CASTLED_KING_BONUS = 35
EARLY_KING_RANK_PENALTY = 45
EARLY_KING_CENTER_FILE_PENALTY = 15
LOST_CASTLING_RIGHT_PENALTY = 30
PIECE_PHASE_VALUES = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

@dataclass(frozen=True)
class GamePhase:
    opening: float
    middlegame: float
    endgame: float


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

KING_MIDDLEGAME_TABLE = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

KING_ENDGAME_TABLE = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

KING_TABLE = KING_MIDDLEGAME_TABLE

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MIDDLEGAME_TABLE,
}

PIECE_SQUARE_PHASE_MULTIPLIERS = {
    chess.PAWN: GamePhase(opening=0.8, middlegame=1.0, endgame=1.4),
    chess.KNIGHT: GamePhase(opening=1.2, middlegame=1.0, endgame=0.8),
    chess.BISHOP: GamePhase(opening=1.1, middlegame=1.0, endgame=0.9),
    chess.ROOK: GamePhase(opening=0.6, middlegame=1.0, endgame=1.2),
    chess.QUEEN: GamePhase(opening=0.7, middlegame=1.0, endgame=1.0),
}


def evaluate(board: chess.Board) -> int:
    """Return a score from the side-to-move perspective."""
    if board.is_checkmate():
        return -CHECKMATE_SCORE

    if board.is_game_over(claim_draw=True):
        return 0

    perspective = board.turn
    phase = phase_weights(board)
    return (
        material_score(board, perspective)
        + piece_square_score(board, perspective, phase)
        + scale_by_phase(
            mobility_score(board, perspective),
            phase,
            GamePhase(opening=0.5, middlegame=1.0, endgame=0.4),
        )
        + scale_by_phase(
            king_safety_score(board, perspective),
            phase,
            GamePhase(opening=1.0, middlegame=0.8, endgame=0.15),
        )
        + king_development_score(board, perspective, phase)
        + scale_by_phase(
            pawn_structure_score(board, perspective),
            phase,
            GamePhase(opening=0.7, middlegame=1.0, endgame=1.4),
        )
        + scale_by_phase(
            bishop_pair_score(board, perspective),
            phase,
            GamePhase(opening=0.8, middlegame=1.0, endgame=1.1),
        )
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


def phase_weights(board: chess.Board) -> GamePhase:
    """Return smooth opening/middlegame/endgame weights for this position."""
    remaining_phase = 0
    for piece_type, phase_value in PIECE_PHASE_VALUES.items():
        remaining_phase += (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        ) * phase_value

    material_ratio = min(1.0, remaining_phase / MAX_MATERIAL_PHASE)
    endgame = 1.0 - material_ratio
    opening_ratio = max(0.0, min(1.0, 1.0 - current_ply(board) / MAX_OPENING_PLY))
    opening = material_ratio * opening_ratio
    middlegame = max(0.0, 1.0 - opening - endgame)
    return GamePhase(opening=opening, middlegame=middlegame, endgame=endgame)


def current_ply(board: chess.Board) -> int:
    if board.move_stack:
        return len(board.move_stack)
    return max(0, (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1))


def scale_by_phase(score: int, phase: GamePhase, multipliers: GamePhase) -> int:
    multiplier = (
        phase.opening * multipliers.opening
        + phase.middlegame * multipliers.middlegame
        + phase.endgame * multipliers.endgame
    )
    return round(score * multiplier)


def piece_square_score(
    board: chess.Board,
    perspective: chess.Color,
    phase: GamePhase | None = None,
) -> int:
    """Return piece square score from `perspective`."""
    if phase is None:
        phase = phase_weights(board)

    white_score = 0
    black_score = 0

    for piece_type in PIECE_VALUES:
        for square in board.pieces(piece_type, chess.WHITE):
            white_score += piece_square_value(piece_type, square, chess.WHITE, phase)

        for square in board.pieces(piece_type, chess.BLACK):
            black_score += piece_square_value(piece_type, square, chess.BLACK, phase)

    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def piece_square_value(
    piece_type: chess.PieceType,
    square: chess.Square,
    color: chess.Color,
    phase: GamePhase,
) -> int:
    oriented_square = square if color == chess.WHITE else chess.square_mirror(square)
    if piece_type == chess.KING:
        middlegame_value = KING_MIDDLEGAME_TABLE[oriented_square]
        endgame_value = KING_ENDGAME_TABLE[oriented_square]
        return round(
            middlegame_value * (phase.opening + phase.middlegame)
            + endgame_value * phase.endgame
        )

    table = PIECE_SQUARE_TABLES[piece_type]
    multipliers = PIECE_SQUARE_PHASE_MULTIPLIERS[piece_type]
    return scale_by_phase(table[oriented_square], phase, multipliers)


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


def king_development_score(
    board: chess.Board,
    perspective: chess.Color,
    phase: GamePhase | None = None,
) -> int:
    if phase is None:
        phase = phase_weights(board)

    white_score = king_development_for_color(board, chess.WHITE, phase)
    black_score = king_development_for_color(board, chess.BLACK, phase)
    score = white_score - black_score
    return score if perspective == chess.WHITE else -score


def king_development_for_color(
    board: chess.Board,
    color: chess.Color,
    phase: GamePhase,
) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0

    pressure = phase.opening + phase.middlegame * 0.6
    if pressure <= 0:
        return 0

    rank = chess.square_rank(king_square)
    file_index = chess.square_file(king_square)
    distance_from_home_rank = rank if color == chess.WHITE else 7 - rank
    score = 0

    if king_square in castled_king_squares(color):
        score += CASTLED_KING_BONUS
    elif distance_from_home_rank > 0:
        score -= EARLY_KING_RANK_PENALTY * distance_from_home_rank
        if file_index in (3, 4, 5):
            score -= EARLY_KING_CENTER_FILE_PENALTY
    elif not has_any_castling_right(board, color):
        score -= LOST_CASTLING_RIGHT_PENALTY

    return round(score * pressure)


def castled_king_squares(color: chess.Color) -> tuple[chess.Square, chess.Square]:
    if color == chess.WHITE:
        return chess.G1, chess.C1
    return chess.G8, chess.C8


def has_any_castling_right(board: chess.Board, color: chess.Color) -> bool:
    return (
        board.has_kingside_castling_rights(color)
        or board.has_queenside_castling_rights(color)
    )


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
        "game phase weights",
        "piece-square tables",
        "mobility",
        "king safety",
        "king development",
        "pawn structure",
        "bishop pair",
    ]
