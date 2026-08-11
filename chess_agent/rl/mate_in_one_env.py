from typing import Any

import chess
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_agent.rl.actions import ACTION_SIZE, action_to_move, legal_action_mask
from chess_agent.rl.observations import OBSERVATION_SHAPE, board_to_observation

DEFAULT_MATE_IN_ONE_FENS = (
    "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/5kq1/8/7K b - - 0 1",
    "6k1/8/6K1/8/8/8/8/R7 w - - 0 1",
    "r7/8/8/8/8/6k1/8/6K1 b - - 0 1",
)


class ChessMateInOneEnv(gym.Env):
    """One-step chess puzzle environment.

    The agent receives one chess position and must choose a legal move that
    checkmates immediately.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(
        self,
        puzzles: tuple[str, ...] | list[str] | None = None,
        *,
        render_mode: str | None = None,
        illegal_action_reward: float = -1.0,
        wrong_move_reward: float = -1.0,
        mate_reward: float = 1.0,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.puzzles = tuple(puzzles or DEFAULT_MATE_IN_ONE_FENS)
        if not self.puzzles:
            raise ValueError("at least one puzzle is required")

        self.render_mode = render_mode
        self.illegal_action_reward = illegal_action_reward
        self.wrong_move_reward = wrong_move_reward
        self.mate_reward = mate_reward
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=1,
                    shape=OBSERVATION_SHAPE,
                    dtype=np.int8,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(ACTION_SIZE,),
                    dtype=np.int8,
                ),
            }
        )
        self.board: chess.Board | None = None
        self.puzzle_index: int | None = None

        for fen in self.puzzles:
            self._validate_puzzle(fen)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        puzzle_index = self._select_puzzle_index(options)
        self.puzzle_index = puzzle_index
        self.board = chess.Board(self.puzzles[puzzle_index])
        return self._observation(), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        board = self._require_board()
        terminated = True
        truncated = False

        try:
            move = action_to_move(int(action))
        except ValueError:
            return (
                self._observation(),
                self.illegal_action_reward,
                terminated,
                truncated,
                self._info(illegal_action=True),
            )

        if move not in board.legal_moves:
            return (
                self._observation(),
                self.illegal_action_reward,
                terminated,
                truncated,
                self._info(illegal_action=True, move_uci=move.uci()),
            )

        move_san = board.san(move)
        board.push(move)
        is_checkmate = board.is_checkmate()
        reward = self.mate_reward if is_checkmate else self.wrong_move_reward
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(
                illegal_action=False,
                is_checkmate=is_checkmate,
                move_san=move_san,
                move_uci=move.uci(),
            ),
        )

    def render(self) -> str | None:
        board = self._require_board()
        rendered = str(board)
        if self.render_mode == "human":
            print(rendered)
            return None
        return rendered

    def action_masks(self) -> np.ndarray:
        return self._action_mask().astype(bool)

    def _observation(self) -> dict[str, np.ndarray]:
        board = self._require_board()
        return {
            "board": board_to_observation(board),
            "action_mask": self._action_mask(),
        }

    def _info(self, **extra: Any) -> dict[str, Any]:
        board = self._require_board()
        info: dict[str, Any] = {
            "fen": board.fen(),
            "puzzle_index": self.puzzle_index,
            "legal_moves": [move.uci() for move in board.legal_moves],
            "action_mask": self._action_mask(),
        }
        info.update(extra)
        return info

    def _action_mask(self) -> np.ndarray:
        return legal_action_mask(self._require_board())

    def _select_puzzle_index(self, options: dict[str, Any] | None) -> int:
        if options is not None and "puzzle_index" in options:
            puzzle_index = int(options["puzzle_index"])
            if not 0 <= puzzle_index < len(self.puzzles):
                raise ValueError(f"puzzle_index out of range: {puzzle_index}")
            return puzzle_index
        return int(self.np_random.integers(len(self.puzzles)))

    def _require_board(self) -> chess.Board:
        if self.board is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.board

    @staticmethod
    def _validate_puzzle(fen: str) -> None:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"invalid puzzle FEN: {fen}")
        if not any(is_mate_after_move(board, move) for move in board.legal_moves):
            raise ValueError(f"puzzle has no mate-in-one move: {fen}")


def is_mate_after_move(board: chess.Board, move: chess.Move) -> bool:
    board_copy = board.copy(stack=False)
    board_copy.push(move)
    return board_copy.is_checkmate()
