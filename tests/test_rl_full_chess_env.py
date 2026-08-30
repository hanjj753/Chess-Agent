import chess
import numpy as np

from chess_agent.agents.base import Agent
from chess_agent.agents.alpha_random_agent import AlphaRandomAgent
from chess_agent.rl.actions import move_to_action
from chess_agent.rl.full_chess_env import FullChessEnv
from chess_agent.rl.observations import (
    OBSERVATION_CHANNELS,
    board_to_observation,
    boards_to_history_observation,
    history_observation_shape,
)


class SequenceAgent(Agent):
    name = "sequence"

    def __init__(self, moves: list[str]) -> None:
        self.moves = list(moves)

    def select_move(self, board: chess.Board) -> chess.Move | None:
        if not self.moves:
            return None
        return chess.Move.from_uci(self.moves.pop(0))


def test_history_observation_stores_current_board_first() -> None:
    initial = chess.Board()
    after_e4 = initial.copy(stack=True)
    after_e4.push_uci("e2e4")

    observation = boards_to_history_observation(
        [initial, after_e4],
        history_length=2,
    )

    assert observation.shape == history_observation_shape(2)
    assert np.array_equal(
        observation[:OBSERVATION_CHANNELS],
        board_to_observation(after_e4),
    )
    assert np.array_equal(
        observation[OBSERVATION_CHANNELS : 2 * OBSERVATION_CHANNELS],
        board_to_observation(initial),
    )
    assert not observation[2 * OBSERVATION_CHANNELS :].any()


def test_full_chess_env_step_plays_agent_and_opponent_moves() -> None:
    env = FullChessEnv(
        opponent=SequenceAgent(["e7e5"]),
        agent_color=chess.WHITE,
        history_length=2,
    )
    observation, info = env.reset()

    assert observation["board"].shape == (54, 8, 8)
    assert observation["action_mask"].sum() == 20
    assert info["agent_color"] == "white"

    observation, reward, terminated, truncated, info = env.step(
        move_to_action(chess.Move.from_uci("e2e4"))
    )

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["opponent_move_uci"] == "e7e5"
    assert info["episode_plies"] == 2
    assert env.board is not None and env.board.turn == chess.WHITE
    assert env.observation_space.contains(observation)

    initial = chess.Board()
    after_e4 = initial.copy(stack=True)
    after_e4.push_uci("e2e4")
    after_e4_e5 = after_e4.copy(stack=True)
    after_e4_e5.push_uci("e7e5")
    assert np.array_equal(
        observation["board"][:OBSERVATION_CHANNELS],
        board_to_observation(after_e4_e5),
    )
    assert np.array_equal(
        observation["board"][OBSERVATION_CHANNELS : 2 * OBSERVATION_CHANNELS],
        board_to_observation(after_e4),
    )


def test_full_chess_env_plays_opening_opponent_move_when_agent_is_black() -> None:
    env = FullChessEnv(
        opponent=SequenceAgent(["e2e4"]),
        agent_color=chess.BLACK,
    )

    observation, info = env.reset()

    assert info["opponent_move_uci"] == "e2e4"
    assert info["episode_plies"] == 1
    assert info["agent_color"] == "black"
    assert env.board is not None and env.board.turn == chess.BLACK
    assert observation["action_mask"].sum() == len(list(env.board.legal_moves))


def test_full_chess_env_reseeds_stochastic_opponent() -> None:
    env = FullChessEnv(
        opponent=AlphaRandomAgent(alpha_move_probability=0.5, depth=1),
        agent_color=chess.BLACK,
    )

    _, first_info = env.reset(seed=123)
    _, repeated_info = env.reset(seed=123)

    assert first_info["opponent_move_uci"] == repeated_info["opponent_move_uci"]


def test_full_chess_env_rewards_agent_checkmate() -> None:
    env = FullChessEnv(
        initial_fen="7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
        agent_color=chess.WHITE,
    )
    env.reset()

    observation, reward, terminated, truncated, info = env.step(
        move_to_action(chess.Move.from_uci("g6g7"))
    )

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["result"] == "1-0"
    assert info["termination"] == "checkmate"
    assert observation["action_mask"].sum() == 0


def test_full_chess_env_truncates_at_max_plies() -> None:
    env = FullChessEnv(
        opponent=SequenceAgent(["e7e5"]),
        agent_color=chess.WHITE,
        max_plies=1,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step(
        move_to_action(chess.Move.from_uci("e2e4"))
    )

    assert reward == 0.0
    assert not terminated
    assert truncated
    assert info["termination"] == "max_plies"
    assert info["episode_plies"] == 1


def test_full_chess_env_terminates_on_illegal_action() -> None:
    env = FullChessEnv(agent_color=chess.WHITE)
    env.reset()

    _, reward, terminated, truncated, info = env.step(
        move_to_action(chess.Move.from_uci("e2e5"))
    )

    assert reward == -1.0
    assert terminated
    assert not truncated
    assert info["illegal_action"]
    assert info["termination"] == "illegal_action"
