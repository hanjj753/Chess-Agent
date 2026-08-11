import chess

from chess_agent.rl.actions import ACTION_SIZE, action_to_move, legal_action_mask, move_to_action


def test_move_action_round_trip_for_normal_move() -> None:
    move = chess.Move.from_uci("e2e4")

    assert action_to_move(move_to_action(move)) == move


def test_move_action_round_trip_for_promotion() -> None:
    move = chess.Move.from_uci("e7e8q")

    assert action_to_move(move_to_action(move)) == move


def test_legal_action_mask_marks_legal_moves() -> None:
    board = chess.Board()
    mask = legal_action_mask(board)

    assert mask.shape == (ACTION_SIZE,)
    assert mask[move_to_action(chess.Move.from_uci("e2e4"))] == 1
    assert mask[move_to_action(chess.Move.from_uci("e2e5"))] == 0
