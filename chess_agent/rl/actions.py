import numpy as np
import chess

PROMOTIONS: tuple[chess.PieceType | None, ...] = (
    None,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)
PROMOTION_TO_INDEX = {promotion: index for index, promotion in enumerate(PROMOTIONS)}
ACTION_SIZE = 64 * 64 * len(PROMOTIONS) # 64칸 중 한 칸 -> 64칸 중 다른 한 칸 이동 * 프로모션 여부


def move_to_action(move: chess.Move) -> int:
    promotion_index = PROMOTION_TO_INDEX.get(move.promotion)
    if promotion_index is None:
        raise ValueError(f"unsupported promotion piece: {move.promotion}")
    return ((move.from_square * 64) + move.to_square) * len(PROMOTIONS) + promotion_index


def action_to_move(action: int) -> chess.Move:
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action out of range: {action}")

    base, promotion_index = divmod(action, len(PROMOTIONS))
    from_square, to_square = divmod(base, 64)
    return chess.Move(
        from_square=from_square,
        to_square=to_square,
        promotion=PROMOTIONS[promotion_index],
    )


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.int8)
    for move in board.legal_moves:
        mask[move_to_action(move)] = 1
    return mask
