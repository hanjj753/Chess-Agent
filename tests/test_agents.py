import chess

import pytest

from chess_agent.agents import AlphaBetaAgent, AlphaRandomAgent, HumanAgent, RandomAgent
from chess_agent.agents.human_agent import parse_human_move


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


def test_alpha_random_agent_is_reproducible_after_seed_reset() -> None:
    board = chess.Board()
    agent = AlphaRandomAgent(alpha_move_probability=0.0)

    agent.reset(seed=42)
    first_move = agent.select_move(board)
    agent.reset(seed=42)
    repeated_move = agent.select_move(board)

    assert first_move == repeated_move
    assert first_move in board.legal_moves


def test_alpha_random_agent_supports_alpha_only_mode() -> None:
    board = chess.Board()
    agent = AlphaRandomAgent(alpha_move_probability=1.0, depth=1)

    assert agent.select_move(board) == AlphaBetaAgent(depth=1).select_move(board)


@pytest.mark.parametrize("probability", (-0.1, 1.1, float("nan")))
def test_alpha_random_agent_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        AlphaRandomAgent(alpha_move_probability=probability)


def test_human_agent_accepts_san_move() -> None:
    board = chess.Board()
    agent = HumanAgent(input_fn=lambda _: "e4", output_fn=lambda _: None)

    assert agent.select_move(board) == chess.Move.from_uci("e2e4")


def test_parse_human_move_accepts_uci_move() -> None:
    board = chess.Board()

    assert parse_human_move(board, "g1f3") == chess.Move.from_uci("g1f3")
