import os
from pathlib import Path
import sys
from typing import Any

import chess
import chess.engine

from chess_agent.agents.base import Agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINES_DIR = PROJECT_ROOT / "engines"
IS_WINDOWS = sys.platform == "win32"
WINDOWS_EXECUTABLE_SUFFIXES = {".exe", ".com", ".bat", ".cmd"}


class UciEngineAgent(Agent):
    """Adapter for external UCI-compatible chess engines."""

    name = "uci"

    def __init__(
        self,
        engine_path: str | Path,
        *,
        time_limit: float | None = 0.1,
        depth: int | None = None,
        nodes: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.engine_path = resolve_engine_path(engine_path)
        self.time_limit = time_limit
        self.depth = depth
        self.nodes = nodes
        self.options = options or {}
        self._engine: chess.engine.SimpleEngine | None = None

    def select_move(self, board: chess.Board) -> chess.Move | None:
        if board.is_game_over(claim_draw=True):
            return None

        limit = chess.engine.Limit(
            time=self.time_limit,
            depth=self.depth,
            nodes=self.nodes,
        )
        result = self._ensure_engine().play(board, limit)
        return result.move

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))
            if self.options:
                self._engine.configure(self.options)
        return self._engine


def resolve_engine_path(
    engine_path: str | Path,
    *,
    engines_dir: Path = DEFAULT_ENGINES_DIR,
) -> Path:
    path = Path(engine_path).expanduser()

    if not path.is_absolute():
        candidate = engines_dir / path
        if candidate.exists():
            path = candidate

    if path.is_dir():
        return find_engine_executable(path)

    if not path.exists():
        raise FileNotFoundError(f"UCI engine not found: {path}")

    if not path.is_file():
        raise ValueError(f"UCI engine must be an executable file: {path}")
    if not IS_WINDOWS:
        if path.suffix.lower() in WINDOWS_EXECUTABLE_SUFFIXES:
            raise ValueError(
                f"Windows engine cannot run natively on Linux/macOS: {path}. "
                "Install a native engine and pass its path with --engine "
                "(for example /usr/games/stockfish on Ubuntu). "
                "chmod +x does not convert a Windows executable."
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"Engine is not executable: {path}. "
                "For a trusted native engine, grant execute permission with chmod +x."
            )
    return path.resolve()


def find_engine_executable(directory: Path) -> Path:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if IS_WINDOWS:
        candidates = [p for p in files if p.suffix.lower() in WINDOWS_EXECUTABLE_SUFFIXES]
    else:
        candidates = [
            p for p in files
            if p.suffix.lower() not in WINDOWS_EXECUTABLE_SUFFIXES
            and os.access(p, os.X_OK)
        ]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple engine executables found in {directory}; "
            "pass the exact executable path instead of the folder: "
            + ", ".join(p.name for p in candidates)
        )
    if candidates:
        return resolve_engine_path(candidates[0].resolve())
    if not IS_WINDOWS and any(p.suffix.lower() in WINDOWS_EXECUTABLE_SUFFIXES for p in files):
        raise ValueError(
            f"No native Linux/macOS engine found in {directory}; "
            "Windows .exe files are not usable. Install a native Stockfish binary "
            "and pass its path (for example --engine /usr/games/stockfish on Ubuntu)."
        )
    raise FileNotFoundError(
        f"No engine executable found in: {directory}. "
        "Pass the native executable path; on Linux/macOS check chmod +x permissions."
    )


def parse_engine_option(raw_option: str) -> tuple[str, str | int | float | bool | None]:
    if "=" not in raw_option:
        raise ValueError("engine option must be written as Name=value")

    name, raw_value = raw_option.split("=", 1)
    name = name.strip()
    raw_value = raw_value.strip()

    if not name:
        raise ValueError("engine option name must not be empty")

    return name, parse_engine_option_value(raw_value)


def parse_engine_options(
    raw_options: list[str] | None,
) -> dict[str, str | int | float | bool | None]:
    options = {}
    for raw_option in raw_options or []:
        name, value = parse_engine_option(raw_option)
        options[name] = value
    return options


def parse_engine_option_value(raw_value: str) -> str | int | float | bool | None:
    lowered = raw_value.lower()

    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value
