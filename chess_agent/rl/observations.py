import numpy as np
import chess

OBSERVATION_SHAPE = (18, 8, 8) # 8*8 체스판 x (6-자기 폰, 나이트, 비숍, 룩, 퀸, 킹 위치)+(6-상대기물위치) + (4-캐슬링 권리) + (2-앙파상, 차례)
OBSERVATION_CHANNELS = OBSERVATION_SHAPE[0]

PIECE_PLANES = {
    chess.Piece(chess.PAWN, chess.WHITE): 0,
    chess.Piece(chess.KNIGHT, chess.WHITE): 1,
    chess.Piece(chess.BISHOP, chess.WHITE): 2,
    chess.Piece(chess.ROOK, chess.WHITE): 3,
    chess.Piece(chess.QUEEN, chess.WHITE): 4,
    chess.Piece(chess.KING, chess.WHITE): 5,
    chess.Piece(chess.PAWN, chess.BLACK): 6,
    chess.Piece(chess.KNIGHT, chess.BLACK): 7,
    chess.Piece(chess.BISHOP, chess.BLACK): 8,
    chess.Piece(chess.ROOK, chess.BLACK): 9,
    chess.Piece(chess.QUEEN, chess.BLACK): 10,
    chess.Piece(chess.KING, chess.BLACK): 11,
}


def board_to_observation(board: chess.Board) -> np.ndarray:
    observation = np.zeros(OBSERVATION_SHAPE, dtype=np.int8)

    for square, piece in board.piece_map().items():
        plane = PIECE_PLANES[piece]
        rank = chess.square_rank(square)
        file_index = chess.square_file(square)
        observation[plane, rank, file_index] = 1

    if board.turn == chess.WHITE:
        observation[12, :, :] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        observation[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        observation[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        observation[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        observation[16, :, :] = 1

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file_index = chess.square_file(board.ep_square)
        observation[17, rank, file_index] = 1

    return observation


def history_observation_shape(history_length: int) -> tuple[int, int, int]:
    """Return the shape for the current board plus previous board positions."""
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    return (OBSERVATION_CHANNELS * (history_length + 1), 8, 8)


def boards_to_history_observation(
    boards: list[chess.Board] | tuple[chess.Board, ...],
    *,
    history_length: int,
) -> np.ndarray:
    """Encode current and previous boards, newest first, with zero padding."""
    shape = history_observation_shape(history_length)
    observation = np.zeros(shape, dtype=np.int8)
    frames = tuple(boards)[-(history_length + 1) :]

    for frame_index, board in enumerate(reversed(frames)):
        start = frame_index * OBSERVATION_CHANNELS
        observation[start : start + OBSERVATION_CHANNELS] = board_to_observation(
            board
        )

    return observation
