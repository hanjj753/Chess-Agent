from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import random
from typing import Protocol

import chess
import numpy as np
from sb3_contrib import MaskablePPO

from chess_agent.agents.base import Agent
from chess_agent.rl.actions import action_to_move, legal_action_mask
from chess_agent.rl.observations import (
    OBSERVATION_CHANNELS,
    boards_to_history_observation,
)


class MaskedPolicyModel(Protocol):
    observation_space: object

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, object]: ...


class FrozenPPOAgent(Agent):
    """Use a saved PPO policy as a read-only chess opponent."""

    def __init__(
        self,
        model: MaskedPolicyModel,
        *,
        name: str = "policy",
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.name = name
        self.deterministic = deterministic
        self.history_length = infer_history_length(model)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> "FrozenPPOAgent":
        resolved_path = Path(path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"PPO opponent checkpoint not found: {path}")
        model = load_frozen_ppo_model(str(resolved_path), device)
        return cls(
            model,
            name=f"policy:{resolved_path.stem}",
            deterministic=deterministic,
        )

    def select_move(self, board: chess.Board) -> chess.Move | None:
        if board.legal_moves.count() == 0:
            return None
        observation = boards_to_history_observation(
            board_history(board, history_length=self.history_length),
            history_length=self.history_length,
        )
        mask = legal_action_mask(board).astype(bool)
        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=mask,
        )
        move = action_to_move(int(np.asarray(action).item()))
        if move not in board.legal_moves:
            raise RuntimeError(
                f"frozen PPO opponent returned an illegal move: {move.uci()}"
            )
        return move


@dataclass(frozen=True)
class OpponentPoolEntry:
    name: str
    agent: Agent
    weight: float


class OpponentPoolAgent(Agent):
    """Choose one weighted opponent at reset and keep it for the whole game."""

    name = "pool"

    def __init__(self, entries: tuple[OpponentPoolEntry, ...]) -> None:
        if not entries:
            raise ValueError("opponent pool requires at least one entry")
        for entry in entries:
            if not math.isfinite(entry.weight) or entry.weight <= 0:
                raise ValueError("opponent pool weights must be positive")
        self.entries = entries
        self._random = random.Random()
        self._active_entry: OpponentPoolEntry | None = None

    @property
    def active_name(self) -> str:
        if self._active_entry is None:
            return self.name
        return self._active_entry.name

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._random.seed(seed)
            for index, entry in enumerate(self.entries):
                entry.agent.reset(seed=seed + index + 1)
        index = self._random.choices(
            range(len(self.entries)),
            weights=[entry.weight for entry in self.entries],
            k=1,
        )[0]
        self._active_entry = self.entries[index]
        if seed is None:
            self._active_entry.agent.reset()

    def select_move(self, board: chess.Board) -> chess.Move | None:
        if self._active_entry is None:
            self.reset()
        assert self._active_entry is not None
        return self._active_entry.agent.select_move(board)


@lru_cache(maxsize=None)
def load_frozen_ppo_model(path: str, device: str) -> MaskablePPO:
    """Load each frozen checkpoint once so vector environments share it."""
    model = MaskablePPO.load(path, device=device)
    model.policy.set_training_mode(False)
    return model


def infer_history_length(model: MaskedPolicyModel) -> int:
    shape = getattr(model.observation_space, "shape", None)
    if shape is None or len(shape) != 3 or tuple(shape[1:]) != (8, 8):
        raise ValueError("PPO opponent does not use a chess board observation")
    channels = int(shape[0])
    if channels % OBSERVATION_CHANNELS != 0:
        raise ValueError("PPO opponent input channels are not a multiple of 18")
    history_length = channels // OBSERVATION_CHANNELS - 1
    if history_length < 0:
        raise ValueError("PPO opponent has an invalid history length")
    return history_length


def board_history(
    board: chess.Board,
    *,
    history_length: int,
) -> tuple[chess.Board, ...]:
    """Reconstruct oldest-to-newest frames from the board's move stack."""
    cursor = board.copy(stack=True)
    newest_first = [cursor.copy(stack=False)]
    for _ in range(history_length):
        if not cursor.move_stack:
            break
        cursor.pop()
        newest_first.append(cursor.copy(stack=False))
    return tuple(reversed(newest_first))
