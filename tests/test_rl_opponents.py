import chess
from gymnasium import spaces
import numpy as np
import pytest

from chess_agent.agents.base import Agent
from chess_agent.rl.actions import move_to_action
from chess_agent.rl.observations import (
    OBSERVATION_CHANNELS,
    board_to_observation,
    history_observation_shape,
)
from chess_agent.rl.opponents import (
    FrozenPPOAgent,
    OpponentPoolAgent,
    OpponentPoolEntry,
    board_history,
)


class FirstMaskedActionModel:
    def __init__(self, history_length: int) -> None:
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=history_observation_shape(history_length),
            dtype=np.int8,
        )
        self.last_observation: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, object]:
        assert deterministic
        self.last_observation = observation.copy()
        self.last_mask = action_masks.copy()
        action = int(np.flatnonzero(action_masks)[0])
        return np.asarray(action), None


class CountingAgent(Agent):
    def __init__(self, name: str) -> None:
        self.name = name
        self.moves = 0

    def select_move(self, board: chess.Board) -> chess.Move | None:
        self.moves += 1
        return next(iter(board.legal_moves), None)


def test_board_history_reconstructs_recent_move_stack() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    board.push_uci("g1f3")

    frames = board_history(board, history_length=2)

    assert len(frames) == 3
    assert frames[0].ply() == 1
    assert frames[1].ply() == 2
    assert frames[2].fen() == board.fen()


def test_frozen_ppo_agent_uses_history_and_legal_action_mask() -> None:
    model = FirstMaskedActionModel(history_length=2)
    agent = FrozenPPOAgent(model)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")

    move = agent.select_move(board)

    assert move in board.legal_moves
    assert model.last_mask is not None
    assert model.last_mask[move_to_action(move)]
    assert model.last_observation is not None
    assert np.array_equal(
        model.last_observation[:OBSERVATION_CHANNELS],
        board_to_observation(board),
    )


def test_frozen_ppo_agent_returns_none_for_terminal_board() -> None:
    model = FirstMaskedActionModel(history_length=1)
    agent = FrozenPPOAgent(model)
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert agent.select_move(board) is None
    assert model.last_observation is None


def test_opponent_pool_keeps_one_agent_for_the_whole_game() -> None:
    first = CountingAgent("first")
    second = CountingAgent("second")
    pool = OpponentPoolAgent(
        (
            OpponentPoolEntry("first", first, 1.0),
            OpponentPoolEntry("second", second, 1.0),
        )
    )
    board = chess.Board()

    pool.reset(seed=17)
    selected_name = pool.active_name
    pool.select_move(board)
    pool.select_move(board)

    selected = first if selected_name == "first" else second
    unselected = second if selected is first else first
    assert selected.moves == 2
    assert unselected.moves == 0

    pool.reset(seed=17)
    assert pool.active_name == selected_name


@pytest.mark.parametrize("weight", (0.0, -1.0, float("nan")))
def test_opponent_pool_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="weights"):
        OpponentPoolAgent(
            (OpponentPoolEntry("invalid", CountingAgent("invalid"), weight),)
        )
