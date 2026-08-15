from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chess
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_agent.rl.actions import ACTION_SIZE, action_to_move, legal_action_mask
from chess_agent.rl.observations import OBSERVATION_SHAPE, board_to_observation


DEFAULT_TACTICAL_PUZZLES = (
    (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ("e7e5", "g1f3", "b8c6"),
        None,
        ("opening",),
    ),
)


@dataclass(frozen=True)
class TacticalPuzzle:
    initial_fen: str
    line_uci: tuple[str, ...]
    rating: int | None = None
    themes: tuple[str, ...] = ()

    @property
    def agent_move_count(self) -> int:
        return (len(self.line_uci) + 1) // 2


class TacticalPuzzleEnv(gym.Env):
    """Multi-step puzzle environment using a fixed tactical line.

    The agent must play the moves at even indices of ``line_uci``. Moves at odd
    indices are forced opponent replies and are applied automatically.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(
        self,
        puzzles: Sequence[TacticalPuzzle] | None = None,
        *,
        puzzles_file: str | Path | None = None,
        render_mode: str | None = None,
        illegal_action_reward: float = -1.0,
        wrong_move_reward: float = -1.0,
        correct_move_reward: float = 0.0,
        success_reward: float = 1.0,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if puzzles is not None and puzzles_file is not None:
            raise ValueError("use either puzzles or puzzles_file, not both")

        self.puzzles = tuple(
            load_tactical_puzzles(puzzles_file)
            if puzzles_file is not None
            else puzzles or default_tactical_puzzles()
        )
        if not self.puzzles:
            raise ValueError("at least one puzzle is required")

        self.render_mode = render_mode
        self.illegal_action_reward = illegal_action_reward
        self.wrong_move_reward = wrong_move_reward
        self.correct_move_reward = correct_move_reward
        self.success_reward = success_reward
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
        self.line_index = 0
        self.correct_agent_moves = 0
        self.done = False

        for puzzle in self.puzzles:
            validate_tactical_puzzle(puzzle)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.puzzle_index = self._select_puzzle_index(options)
        puzzle = self._current_puzzle()
        self.board = chess.Board(puzzle.initial_fen)
        self.line_index = 0
        self.correct_agent_moves = 0
        self.done = False
        return self._observation(), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        board = self._require_board()
        puzzle = self._current_puzzle()
        terminated = True
        truncated = False

        if self.done:
            raise RuntimeError("reset() must be called before stepping again")

        try:
            move = action_to_move(int(action))
        except ValueError:
            self.done = True
            return (
                self._observation(),
                self.illegal_action_reward,
                terminated,
                truncated,
                self._info(illegal_action=True),
            )

        expected_move = chess.Move.from_uci(puzzle.line_uci[self.line_index])
        if move not in board.legal_moves:
            self.done = True
            return (
                self._observation(),
                self.illegal_action_reward,
                terminated,
                truncated,
                self._info(
                    illegal_action=True,
                    expected_move_uci=expected_move.uci(),
                    move_uci=move.uci(),
                ),
            )

        move_san = board.san(move)
        if move != expected_move:
            self.done = True
            return (
                self._observation(),
                self.wrong_move_reward,
                terminated,
                truncated,
                self._info(
                    illegal_action=False,
                    is_correct=False,
                    expected_move_uci=expected_move.uci(),
                    move_san=move_san,
                    move_uci=move.uci(),
                ),
            )

        board.push(move)
        self.line_index += 1
        self.correct_agent_moves += 1
        if self.line_index >= len(puzzle.line_uci):
            self.done = True
            return (
                self._observation(),
                self.success_reward,
                terminated,
                truncated,
                self._info(
                    illegal_action=False,
                    is_correct=True,
                    is_success=True,
                    move_san=move_san,
                    move_uci=move.uci(),
                ),
            )

        opponent_reply = chess.Move.from_uci(puzzle.line_uci[self.line_index])
        opponent_reply_san = board.san(opponent_reply)
        board.push(opponent_reply)
        self.line_index += 1

        if self.line_index >= len(puzzle.line_uci):
            self.done = True
            return (
                self._observation(),
                self.success_reward,
                terminated,
                truncated,
                self._info(
                    illegal_action=False,
                    is_correct=True,
                    is_success=True,
                    move_san=move_san,
                    move_uci=move.uci(),
                    opponent_reply_san=opponent_reply_san,
                    opponent_reply_uci=opponent_reply.uci(),
                ),
            )

        terminated = False
        return (
            self._observation(),
            self.correct_move_reward,
            terminated,
            truncated,
            self._info(
                illegal_action=False,
                is_correct=True,
                is_success=False,
                move_san=move_san,
                move_uci=move.uci(),
                opponent_reply_san=opponent_reply_san,
                opponent_reply_uci=opponent_reply.uci(),
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
        puzzle = self._current_puzzle()
        expected_move = None
        if not self.done and self.line_index < len(puzzle.line_uci):
            expected_move = puzzle.line_uci[self.line_index]

        info: dict[str, Any] = {
            "fen": board.fen(),
            "puzzle_index": self.puzzle_index,
            "line_index": self.line_index,
            "expected_move_uci": expected_move,
            "correct_agent_moves": self.correct_agent_moves,
            "total_agent_moves": puzzle.agent_move_count,
            "legal_moves": [move.uci() for move in board.legal_moves],
            "action_mask": self._action_mask(),
            "rating": puzzle.rating,
            "themes": puzzle.themes,
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

    def _current_puzzle(self) -> TacticalPuzzle:
        if self.puzzle_index is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.puzzles[self.puzzle_index]

    def _require_board(self) -> chess.Board:
        if self.board is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.board


def default_tactical_puzzles() -> tuple[TacticalPuzzle, ...]:
    return tuple(
        TacticalPuzzle(
            initial_fen=fen,
            line_uci=tuple(line_uci),
            rating=rating,
            themes=tuple(themes),
        )
        for fen, line_uci, rating, themes in DEFAULT_TACTICAL_PUZZLES
    )


def validate_tactical_puzzle(puzzle: TacticalPuzzle) -> None:
    if not puzzle.line_uci:
        raise ValueError(f"puzzle has empty tactical line: {puzzle.initial_fen}")

    board = chess.Board(puzzle.initial_fen)
    if not board.is_valid():
        raise ValueError(f"invalid puzzle FEN: {puzzle.initial_fen}")

    for move_uci in puzzle.line_uci:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise ValueError(f"invalid UCI move in tactical line: {move_uci}") from exc
        if move not in board.legal_moves:
            raise ValueError(
                f"illegal move in tactical line: {move_uci} from {board.fen()}"
            )
        board.push(move)


def load_tactical_puzzles(path: str | Path) -> tuple[TacticalPuzzle, ...]:
    puzzles = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        puzzles.append(parse_tactical_puzzle_line(line, line_number=line_number))

    if not puzzles:
        raise ValueError(f"no tactical puzzles found in: {path}")
    return tuple(puzzles)


def parse_tactical_puzzle_line(line: str, *, line_number: int) -> TacticalPuzzle:
    parts = line.split("\t")
    if len(parts) < 2:
        raise ValueError(f"expected FEN and tactical line at line {line_number}")

    fen = parts[0].strip()
    line_uci = tuple(parts[1].split())
    rating = parse_optional_int(parts[2].strip()) if len(parts) >= 3 else None
    themes = tuple(parts[3].split()) if len(parts) >= 4 and parts[3].strip() else ()
    puzzle = TacticalPuzzle(
        initial_fen=fen,
        line_uci=line_uci,
        rating=rating,
        themes=themes,
    )
    try:
        validate_tactical_puzzle(puzzle)
    except ValueError as exc:
        raise ValueError(f"invalid tactical puzzle at line {line_number}: {exc}") from exc
    return puzzle


def parse_optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def format_tactical_puzzle_line(puzzle: TacticalPuzzle) -> str:
    rating = "" if puzzle.rating is None else str(puzzle.rating)
    themes = " ".join(puzzle.themes)
    return f"{puzzle.initial_fen}\t{' '.join(puzzle.line_uci)}\t{rating}\t{themes}\n"
