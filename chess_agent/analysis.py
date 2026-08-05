from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

from chess_agent.agents.uci_engine_agent import resolve_engine_path

MATE_SCORE = 100_000


@dataclass(frozen=True)
class MoveAnalysis:
    ply: int
    move_number: int
    color: str
    san: str
    uci: str
    fen_before: str
    fen_after: str
    score_before_cp: int
    score_after_cp: int
    loss_cp: int
    best_move_uci: str | None
    best_move_san: str | None
    label: str


def analyze_pgn(
    *,
    pgn_path: str | Path,
    engine_path: str | Path,
    time_limit: float | None = 0.1,
    depth: int | None = None,
    nodes: int | None = None,
    options: dict[str, Any] | None = None,
) -> list[MoveAnalysis]:
    game = read_first_game(pgn_path)
    board = game.board()
    analyses: list[MoveAnalysis] = []
    engine = chess.engine.SimpleEngine.popen_uci(str(resolve_engine_path(engine_path)))

    try:
        if options:
            engine.configure(options)

        limit = chess.engine.Limit(time=time_limit, depth=depth, nodes=nodes)

        for ply, move in enumerate(game.mainline_moves(), start=1):
            color = board.turn
            fen_before = board.fen()
            san = board.san(move)
            info_before = engine.analyse(board, limit)
            score_before = info_score_cp(info_before, color)
            best_move = best_move_from_info(info_before)
            best_move_san = (
                board.san(best_move)
                if best_move is not None and best_move in board.legal_moves
                else None
            )

            board.push(move)

            info_after = engine.analyse(board, limit)
            score_after = info_score_cp(info_after, color)
            loss_cp = max(0, score_before - score_after)

            analyses.append(
                MoveAnalysis(
                    ply=ply,
                    move_number=(ply + 1) // 2,
                    color=color_name(color),
                    san=san,
                    uci=move.uci(),
                    fen_before=fen_before,
                    fen_after=board.fen(),
                    score_before_cp=score_before,
                    score_after_cp=score_after,
                    loss_cp=loss_cp,
                    best_move_uci=best_move.uci() if best_move is not None else None,
                    best_move_san=best_move_san,
                    label=classify_loss(loss_cp),
                )
            )
    finally:
        engine.quit()

    return analyses


def read_first_game(pgn_path: str | Path) -> chess.pgn.Game:
    path = Path(pgn_path)
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)

    if game is None:
        raise ValueError(f"No PGN game found in: {path}")
    return game


def best_move_from_info(info: dict[str, Any]) -> chess.Move | None:
    pv = info.get("pv")
    if not pv:
        return None
    return pv[0]


def info_score_cp(info: dict[str, Any], perspective: chess.Color) -> int:
    score = info["score"].pov(perspective).score(mate_score=MATE_SCORE)
    return int(score) if score is not None else 0


def classify_loss(loss_cp: int) -> str:
    if loss_cp >= 300:
        return "blunder"
    if loss_cp >= 150:
        return "mistake"
    if loss_cp >= 75:
        return "inaccuracy"
    return "ok"


def color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def save_analysis_json(analyses: list[MoveAnalysis], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in analyses], indent=2),
        encoding="utf-8",
    )
    return output_path


def load_analysis_json(path: str | Path) -> list[MoveAnalysis]:
    raw_items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MoveAnalysis(**item) for item in raw_items]


def default_analysis_path(pgn_path: str | Path) -> Path:
    path = Path(pgn_path)
    return path.with_suffix(path.suffix + ".analysis.json")


def suspicious_moves(
    analyses: list[MoveAnalysis],
    *,
    threshold_cp: int = 75,
) -> list[MoveAnalysis]:
    return [item for item in analyses if item.loss_cp >= threshold_cp]
