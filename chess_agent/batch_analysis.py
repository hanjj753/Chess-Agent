from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from chess_agent.analysis import (
    MoveAnalysis,
    analyze_pgn,
    default_analysis_path,
    load_analysis_json,
    save_analysis_json,
)

LOSS_CAP_CP = 1_000
MATE_LIKE_LOSS_CP = 50_000


@dataclass(frozen=True)
class GameAnalysis:
    pgn_path: Path
    analysis_path: Path
    moves: list[MoveAnalysis]
    agent_color: str | None = None


@dataclass(frozen=True)
class MoveReference:
    pgn_path: Path
    move: MoveAnalysis


@dataclass(frozen=True)
class PhaseStats:
    name: str
    move_count: int
    average_loss_cp: float
    capped_average_loss_cp: float
    inaccuracies: int
    mistakes: int
    blunders: int
    mate_like_losses: int


@dataclass(frozen=True)
class BatchSummary:
    games: list[GameAnalysis]
    total_moves: int
    average_loss_cp: float
    capped_average_loss_cp: float
    inaccuracies: int
    mistakes: int
    blunders: int
    mate_like_losses: int
    top_losses: list[MoveReference]
    phase_stats: list[PhaseStats]
    agent_only: bool


PHASES = [
    ("opening", 1, 10),
    ("middlegame", 11, 30),
    ("endgame", 31, 10_000),
]


def analyze_folder(
    *,
    folder: str | Path,
    engine_path: str | Path | None,
    time_limit: float | None = 0.1,
    depth: int | None = None,
    nodes: int | None = None,
    options: dict[str, Any] | None = None,
    reuse_existing: bool = True,
    agent_only: bool = True,
    loss_cap_cp: int = LOSS_CAP_CP,
) -> BatchSummary:
    pgn_paths = find_pgn_files(folder)
    games = [
        analyze_or_load_game(
            pgn_path=pgn_path,
            engine_path=engine_path,
            time_limit=time_limit,
            depth=depth,
            nodes=nodes,
            options=options,
            reuse_existing=reuse_existing,
        )
        for pgn_path in pgn_paths
    ]
    return summarize_games(
        games,
        agent_only=agent_only,
        loss_cap_cp=loss_cap_cp,
    )


def find_pgn_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"PGN folder does not exist: {root}")
    return sorted(root.glob("*.pgn"))


def analyze_or_load_game(
    *,
    pgn_path: Path,
    engine_path: str | Path | None,
    time_limit: float | None,
    depth: int | None,
    nodes: int | None,
    options: dict[str, Any] | None,
    reuse_existing: bool,
) -> GameAnalysis:
    analysis_path = default_analysis_path(pgn_path)

    if reuse_existing and analysis_path.exists():
        return GameAnalysis(
            pgn_path=pgn_path,
            analysis_path=analysis_path,
            moves=load_analysis_json(analysis_path),
            agent_color=infer_agent_color(pgn_path),
        )

    if engine_path is None:
        raise ValueError(
            f"Missing engine path and no cached analysis exists for: {pgn_path}"
        )

    moves = analyze_pgn(
        pgn_path=pgn_path,
        engine_path=engine_path,
        time_limit=time_limit,
        depth=depth,
        nodes=nodes,
        options=options,
    )
    save_analysis_json(moves, analysis_path)
    return GameAnalysis(
        pgn_path=pgn_path,
        analysis_path=analysis_path,
        moves=moves,
        agent_color=infer_agent_color(pgn_path),
    )


def summarize_games(
    games: list[GameAnalysis],
    *,
    agent_only: bool = True,
    loss_cap_cp: int = LOSS_CAP_CP,
) -> BatchSummary:
    all_references = references_for_games(games, agent_only=agent_only)
    total_moves = len(all_references)
    average_loss = average_loss_cp([reference.move for reference in all_references])
    capped_average_loss = capped_average_loss_cp(
        [reference.move for reference in all_references],
        cap_cp=loss_cap_cp,
    )
    top_losses = sorted(
        all_references,
        key=lambda reference: reference.move.loss_cp,
        reverse=True,
    )[:10]

    return BatchSummary(
        games=games,
        total_moves=total_moves,
        average_loss_cp=average_loss,
        capped_average_loss_cp=capped_average_loss,
        inaccuracies=count_label(all_references, "inaccuracy"),
        mistakes=count_label(all_references, "mistake"),
        blunders=count_label(all_references, "blunder"),
        mate_like_losses=count_mate_like_losses(all_references),
        top_losses=top_losses,
        phase_stats=phase_summaries(all_references, loss_cap_cp=loss_cap_cp),
        agent_only=agent_only,
    )


def references_for_games(
    games: list[GameAnalysis],
    *,
    agent_only: bool,
) -> list[MoveReference]:
    references = []
    for game in games:
        for move in game.moves:
            if not agent_only or game.agent_color is None or move.color == game.agent_color:
                references.append(MoveReference(pgn_path=game.pgn_path, move=move))
    return references


def average_loss_cp(moves: list[MoveAnalysis]) -> float:
    if not moves:
        return 0.0
    return sum(move.loss_cp for move in moves) / len(moves)


def capped_average_loss_cp(
    moves: list[MoveAnalysis],
    *,
    cap_cp: int = LOSS_CAP_CP,
) -> float:
    if not moves:
        return 0.0
    return sum(min(move.loss_cp, cap_cp) for move in moves) / len(moves)


def count_label(references: list[MoveReference], label: str) -> int:
    return sum(1 for reference in references if reference.move.label == label)


def count_mate_like_losses(references: list[MoveReference]) -> int:
    return sum(1 for reference in references if reference.move.loss_cp >= MATE_LIKE_LOSS_CP)


def phase_summaries(
    references: list[MoveReference],
    *,
    loss_cap_cp: int = LOSS_CAP_CP,
) -> list[PhaseStats]:
    summaries = []

    for name, start_move, end_move in PHASES:
        moves = [
            reference.move
            for reference in references
            if start_move <= reference.move.move_number <= end_move
        ]
        summaries.append(
            PhaseStats(
                name=name,
                move_count=len(moves),
                average_loss_cp=average_loss_cp(moves),
                capped_average_loss_cp=capped_average_loss_cp(
                    moves,
                    cap_cp=loss_cap_cp,
                ),
                inaccuracies=sum(1 for move in moves if move.label == "inaccuracy"),
                mistakes=sum(1 for move in moves if move.label == "mistake"),
                blunders=sum(1 for move in moves if move.label == "blunder"),
                mate_like_losses=sum(1 for move in moves if move.loss_cp >= MATE_LIKE_LOSS_CP),
            )
        )

    return summaries


def infer_agent_color(pgn_path: str | Path) -> str | None:
    match = re.search(r"_game_\d+_(white|black)_", Path(pgn_path).name)
    if match is None:
        return None
    return match.group(1)
