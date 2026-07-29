import chess

from chess_agent.agents import AlphaBetaAgent, RandomAgent


def test_random_agent_returns_legal_move() -> None:
    board = chess.Board()
    move = RandomAgent().select_move(board)

    assert move in board.legal_moves


def test_alpha_beta_agent_returns_legal_move() -> None:
    board = chess.Board()
    move = AlphaBetaAgent(depth=1).select_move(board)

    assert move in board.legal_moves


def test_alpha_beta_agent_returns_none_when_game_is_over() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert board.is_checkmate()
    assert AlphaBetaAgent(depth=1).select_move(board) is None
