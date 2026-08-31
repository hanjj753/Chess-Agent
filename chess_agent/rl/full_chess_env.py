import math
from pathlib import Path
from typing import Any

import chess
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_agent.agents.base import Agent
from chess_agent.agents.random_agent import RandomAgent
from chess_agent.engine.evaluation import evaluate
from chess_agent.rl.actions import ACTION_SIZE, action_to_move, legal_action_mask
from chess_agent.rl.observations import (
    boards_to_history_observation,
    history_observation_shape,
)


class FullChessEnv(gym.Env):
    """A full-game environment where one step contains agent and opponent moves."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(
        self,
        opponent: Agent | None = None,
        *,
        initial_fen: str = chess.STARTING_FEN,
        agent_color: chess.Color | None = None,
        history_length: int = 4,
        max_plies: int = 300,
        render_mode: str | None = None,
        win_reward: float = 1.0,
        draw_reward: float = 0.0,
        loss_reward: float = -1.0,
        illegal_action_reward: float = -1.0,
        reward_shaping_coefficient: float = 0.0,
        reward_shaping_scale: float = 600.0,
        reward_shaping_gamma: float = 0.995,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if history_length < 0:
            raise ValueError("history_length must be non-negative")
        if max_plies < 1:
            raise ValueError("max_plies must be positive")
        if not math.isfinite(reward_shaping_coefficient) or (
            reward_shaping_coefficient < 0
        ):
            raise ValueError("reward_shaping_coefficient must be non-negative")
        if not math.isfinite(reward_shaping_scale) or reward_shaping_scale <= 0:
            raise ValueError("reward_shaping_scale must be positive")
        if not math.isfinite(reward_shaping_gamma) or not (
            0.0 <= reward_shaping_gamma <= 1.0
        ):
            raise ValueError("reward_shaping_gamma must be between 0 and 1")
        validate_initial_fen(initial_fen)

        self.opponent = opponent or RandomAgent()
        self.initial_fen = initial_fen
        self.fixed_agent_color = agent_color
        self.history_length = history_length
        self.max_plies = max_plies
        self.render_mode = render_mode
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward
        self.illegal_action_reward = illegal_action_reward
        self.reward_shaping_coefficient = reward_shaping_coefficient
        self.reward_shaping_scale = reward_shaping_scale
        self.reward_shaping_gamma = reward_shaping_gamma

        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=1,
                    shape=history_observation_shape(history_length),
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
        self.agent_color: chess.Color | None = None
        self.episode_plies = 0
        self.done = False
        self._board_history: list[chess.Board] = []
        self._position_potential = 0.0
        self._episode_extrinsic_reward = 0.0
        self._episode_shaping_reward = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        initial_fen = str(options.get("fen", self.initial_fen))
        validate_initial_fen(initial_fen)

        self.board = chess.Board(initial_fen)
        self.agent_color = self._select_agent_color(options)
        self.episode_plies = 0
        self.done = False
        self._board_history = [self.board.copy(stack=True)]
        self._episode_extrinsic_reward = 0.0
        self._episode_shaping_reward = 0.0
        self.opponent.reset(seed=seed)

        opening_move = None
        if not self.board.is_game_over(claim_draw=True) and self.board.turn != self.agent_color:
            opening_move = self._play_opponent_move()

        if self.board.is_game_over(claim_draw=True):
            self.done = True
        self._position_potential = (
            0.0 if self.done else self._current_position_potential()
        )

        return self._observation(), self._info(opponent_move_uci=opening_move)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        board = self._require_board()
        agent_color = self._require_agent_color()
        if self.done:
            raise RuntimeError("reset() must be called before stepping again")
        if board.turn != agent_color:
            raise RuntimeError("environment step requested outside the agent turn")

        try:
            move = action_to_move(int(action))
        except (TypeError, ValueError):
            self.done = True
            reward, reward_info = self._compose_reward(
                self.illegal_action_reward,
                true_terminal=True,
            )
            return (
                self._observation(),
                reward,
                True,
                False,
                self._info(
                    illegal_action=True,
                    termination="illegal_action",
                    **reward_info,
                ),
            )

        if move not in board.legal_moves:
            self.done = True
            reward, reward_info = self._compose_reward(
                self.illegal_action_reward,
                true_terminal=True,
            )
            return (
                self._observation(),
                reward,
                True,
                False,
                self._info(
                    illegal_action=True,
                    move_uci=move.uci(),
                    termination="illegal_action",
                    **reward_info,
                ),
            )

        move_san = board.san(move)
        self._push(move)
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return self._finish(
                outcome=outcome,
                move_uci=move.uci(),
                move_san=move_san,
            )
        if self.episode_plies >= self.max_plies:
            return self._truncate(move_uci=move.uci(), move_san=move_san)

        opponent_move = self._play_opponent_move()
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return self._finish(
                outcome=outcome,
                move_uci=move.uci(),
                move_san=move_san,
                opponent_move_uci=opponent_move,
            )
        if self.episode_plies >= self.max_plies:
            return self._truncate(
                move_uci=move.uci(),
                move_san=move_san,
                opponent_move_uci=opponent_move,
            )

        reward, reward_info = self._compose_reward(0.0, true_terminal=False)
        return (
            self._observation(),
            reward,
            False,
            False,
            self._info(
                illegal_action=False,
                move_uci=move.uci(),
                move_san=move_san,
                opponent_move_uci=opponent_move,
                **reward_info,
            ),
        )

    def render(self) -> str | None:
        rendered = str(self._require_board())
        if self.render_mode == "human":
            print(rendered)
            return None
        return rendered

    def action_masks(self) -> np.ndarray:
        return self._action_mask().astype(bool)

    def _select_agent_color(self, options: dict[str, Any]) -> chess.Color:
        if "agent_color" in options:
            color = options["agent_color"]
            if color not in (chess.WHITE, chess.BLACK):
                raise ValueError("agent_color must be chess.WHITE or chess.BLACK")
            return color
        if self.fixed_agent_color is not None:
            return self.fixed_agent_color
        return chess.WHITE if int(self.np_random.integers(2)) else chess.BLACK

    def _play_opponent_move(self) -> str | None:
        board = self._require_board()
        if isinstance(self.opponent, RandomAgent):
            legal_moves = list(board.legal_moves)
            move = (
                legal_moves[int(self.np_random.integers(len(legal_moves)))]
                if legal_moves
                else None
            )
        else:
            move = self.opponent.select_move(board)
        if move is None:
            if board.legal_moves.count() == 0:
                return None
            raise RuntimeError("opponent returned no move in a non-terminal position")
        if move not in board.legal_moves:
            raise RuntimeError(f"opponent returned an illegal move: {move.uci()}")
        move_uci = move.uci()
        self._push(move)
        return move_uci

    def _push(self, move: chess.Move) -> None:
        board = self._require_board()
        board.push(move)
        self.episode_plies += 1
        self._board_history.append(board.copy(stack=True))
        max_frames = self.history_length + 1
        if len(self._board_history) > max_frames:
            del self._board_history[:-max_frames]

    def _finish(
        self,
        *,
        outcome: chess.Outcome,
        **extra: Any,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.done = True
        reward, reward_info = self._compose_reward(
            self._outcome_reward(outcome),
            true_terminal=True,
        )
        return (
            self._observation(),
            reward,
            True,
            False,
            self._info(
                illegal_action=False,
                result=outcome.result(),
                termination=outcome.termination.name.lower(),
                **reward_info,
                **extra,
            ),
        )

    def _truncate(
        self,
        **extra: Any,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.done = True
        reward, reward_info = self._compose_reward(
            self.draw_reward,
            true_terminal=False,
        )
        return (
            self._observation(),
            reward,
            False,
            True,
            self._info(
                illegal_action=False,
                result="1/2-1/2",
                termination="max_plies",
                **reward_info,
                **extra,
            ),
        )

    def _outcome_reward(self, outcome: chess.Outcome) -> float:
        if outcome.winner is None:
            return self.draw_reward
        if outcome.winner == self._require_agent_color():
            return self.win_reward
        return self.loss_reward

    def _compose_reward(
        self,
        extrinsic_reward: float,
        *,
        true_terminal: bool,
    ) -> tuple[float, dict[str, float]]:
        previous_potential = self._position_potential
        next_potential = (
            0.0 if true_terminal else self._current_position_potential()
        )
        shaping_reward = self.reward_shaping_coefficient * (
            self.reward_shaping_gamma * next_potential - previous_potential
        )
        training_reward = float(extrinsic_reward) + shaping_reward
        self._position_potential = next_potential
        self._episode_extrinsic_reward += float(extrinsic_reward)
        self._episode_shaping_reward += shaping_reward
        return training_reward, {
            "extrinsic_reward": float(extrinsic_reward),
            "shaping_reward": shaping_reward,
            "training_reward": training_reward,
        }

    def _current_position_potential(self) -> float:
        if self.reward_shaping_coefficient == 0:
            return 0.0
        return normalized_evaluation_potential(
            self._require_board(),
            perspective=self._require_agent_color(),
            scale=self.reward_shaping_scale,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "board": boards_to_history_observation(
                self._board_history,
                history_length=self.history_length,
            ),
            "action_mask": self._action_mask(),
        }

    def _action_mask(self) -> np.ndarray:
        board = self._require_board()
        if self.done or board.turn != self._require_agent_color():
            return np.zeros(ACTION_SIZE, dtype=np.int8)
        return legal_action_mask(board)

    def _info(self, **extra: Any) -> dict[str, Any]:
        board = self._require_board()
        color = self._require_agent_color()
        info: dict[str, Any] = {
            "fen": board.fen(),
            "agent_color": "white" if color == chess.WHITE else "black",
            "episode_plies": self.episode_plies,
            "history_frames": len(self._board_history),
            "legal_moves": (
                [move.uci() for move in board.legal_moves]
                if not self.done and board.turn == color
                else []
            ),
            "result": "*",
            "termination": None,
            "position_potential": self._position_potential,
            "episode_extrinsic_reward": self._episode_extrinsic_reward,
            "episode_shaping_reward": self._episode_shaping_reward,
            "episode_training_reward": (
                self._episode_extrinsic_reward + self._episode_shaping_reward
            ),
        }
        info.update(extra)
        return info

    def _require_board(self) -> chess.Board:
        if self.board is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.board

    def _require_agent_color(self) -> chess.Color:
        if self.agent_color is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.agent_color


def normalized_evaluation_potential(
    board: chess.Board,
    *,
    perspective: chess.Color,
    scale: float,
) -> float:
    """Return the static evaluation normalized to [-1, 1]."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    score = evaluate(board)
    if board.turn != perspective:
        score = -score
    return math.tanh(score / scale)


class BoardOnlyObservation(gym.ObservationWrapper):
    """Expose only board planes while keeping action_masks() for MaskablePPO."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("BoardOnlyObservation requires a Dict observation space")
        self.observation_space = env.observation_space["board"]

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["board"]

    def action_masks(self) -> np.ndarray:
        action_masks = getattr(self.env.unwrapped, "action_masks", None)
        if action_masks is None:
            raise AttributeError("wrapped environment does not provide action_masks()")
        return action_masks()


def validate_initial_fen(fen: str | Path) -> None:
    try:
        board = chess.Board(str(fen))
    except ValueError as exc:
        raise ValueError(f"invalid initial FEN: {fen}") from exc
    if not board.is_valid():
        raise ValueError(f"invalid initial FEN: {fen}")
