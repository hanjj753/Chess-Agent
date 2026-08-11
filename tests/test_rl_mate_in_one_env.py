import chess
import numpy as np

from chess_agent.rl.actions import move_to_action
from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv


def test_mate_in_one_env_reset_returns_observation_and_mask() -> None:
    env = ChessMateInOneEnv()

    observation, info = env.reset(seed=1, options={"puzzle_index": 0})

    assert observation["board"].shape == (18, 8, 8)
    assert observation["action_mask"].shape == (env.action_space.n,)
    assert observation["action_mask"].sum() == len(info["legal_moves"])
    assert env.observation_space.contains(observation)


def test_mate_in_one_env_rewards_checkmate_move() -> None:
    env = ChessMateInOneEnv()
    env.reset(options={"puzzle_index": 0})
    action = move_to_action(chess.Move.from_uci("g6g7"))

    _, reward, terminated, truncated, info = env.step(action)

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["is_checkmate"]
    assert info["move_san"] == "Qg7#"


def test_mate_in_one_env_penalizes_legal_non_mate_move() -> None:
    env = ChessMateInOneEnv()
    env.reset(options={"puzzle_index": 0})
    action = move_to_action(chess.Move.from_uci("f6e6"))

    _, reward, terminated, _, info = env.step(action)

    assert reward == -1.0
    assert terminated
    assert not info["is_checkmate"]


def test_mate_in_one_env_penalizes_illegal_action() -> None:
    env = ChessMateInOneEnv()
    env.reset(options={"puzzle_index": 0})
    action = move_to_action(chess.Move.from_uci("a1a2"))

    _, reward, terminated, _, info = env.step(action)

    assert reward == -1.0
    assert terminated
    assert info["illegal_action"]


def test_mate_in_one_env_action_masks_are_boolean_for_maskable_rl() -> None:
    env = ChessMateInOneEnv()
    observation, _ = env.reset(options={"puzzle_index": 0})

    assert env.action_masks().dtype == np.bool_
    assert np.array_equal(env.action_masks(), observation["action_mask"].astype(bool))
