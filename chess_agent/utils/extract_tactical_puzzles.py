import argparse
import contextlib
import csv
from dataclasses import dataclass
from pathlib import Path
import random

import chess

from chess_agent.rl.tactical_puzzle_env import (
    TacticalPuzzle,
    format_tactical_puzzle_line,
    validate_tactical_puzzle,
)
from chess_agent.utils.extract_mate_in_one_puzzles import (
    OutputHandle,
    open_output_writers,
    open_puzzle_csv,
    parse_optional_int,
    rating_out_of_range,
    select_writer,
)


DEFAULT_THEMES = (
    "mateIn2",
    "mateIn3",
    "fork",
    "pin",
    "skewer",
    "sacrifice",
    "discoveredAttack",
    "deflection",
    "attraction",
    "clearance",
    "intermezzo",
    "trappedPiece",
    "xRayAttack",
)
DEFAULT_EXCLUDED_THEMES = ("mateIn1",)


@dataclass(frozen=True)
class TacticalExtractionStats:
    rows_seen: int = 0
    theme_matches: int = 0
    rating_skips: int = 0
    length_skips: int = 0
    invalid_skips: int = 0
    duplicate_skips: int = 0
    written: int = 0
    train_written: int = 0
    validation_written: int = 0


def extract_tactical_puzzles(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    train_output_path: str | Path | None = None,
    validation_output_path: str | Path | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
    limit: int | None = None,
    themes: tuple[str, ...] | list[str] | set[str] | None = DEFAULT_THEMES,
    exclude_themes: tuple[str, ...] | list[str] | set[str] = DEFAULT_EXCLUDED_THEMES,
    min_agent_moves: int = 2,
    max_agent_moves: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    deduplicate: bool = True,
) -> TacticalExtractionStats:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if min_agent_moves < 1:
        raise ValueError("min_agent_moves must be positive")
    if max_agent_moves is not None and max_agent_moves < min_agent_moves:
        raise ValueError("max_agent_moves must be at least min_agent_moves")

    split_outputs = train_output_path is not None or validation_output_path is not None
    if split_outputs:
        if train_output_path is None or validation_output_path is None:
            raise ValueError("train_output_path and validation_output_path must be used together")
        if output_path is not None:
            raise ValueError("use output_path or train/validation output paths, not both")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
    elif output_path is None:
        raise ValueError("output_path is required when not writing split outputs")

    include_theme_set = set(themes) if themes is not None else None
    exclude_theme_set = set(exclude_themes)
    stats = TacticalExtractionStats()
    seen_keys: set[str] = set()
    rng = random.Random(seed)

    with (
        open_puzzle_csv(input_path) as handle,
        contextlib.ExitStack() as stack,
    ):
        writers = open_output_writers(
            stack=stack,
            output_path=output_path,
            train_output_path=train_output_path,
            validation_output_path=validation_output_path,
        )
        reader = csv.DictReader(handle)
        for row in reader:
            stats = increment(stats, "rows_seen")
            row_themes = set((row.get("Themes") or "").split())
            if not themes_match(row_themes, include_theme_set, exclude_theme_set):
                continue

            stats = increment(stats, "theme_matches")
            rating = parse_optional_int(row.get("Rating"))
            if rating_out_of_range(rating, min_rating, max_rating):
                stats = increment(stats, "rating_skips")
                continue

            puzzle = tactical_puzzle_from_lichess_row(row)
            if puzzle is None:
                stats = increment(stats, "invalid_skips")
                continue
            if not agent_move_count_in_range(
                puzzle.agent_move_count,
                min_agent_moves=min_agent_moves,
                max_agent_moves=max_agent_moves,
            ):
                stats = increment(stats, "length_skips")
                continue

            key = f"{puzzle.initial_fen}\t{' '.join(puzzle.line_uci)}"
            if deduplicate and key in seen_keys:
                stats = increment(stats, "duplicate_skips")
                continue

            seen_keys.add(key)
            target = select_writer(
                writers,
                rng=rng,
                validation_fraction=validation_fraction,
            )
            write_puzzle(target, puzzle)
            stats = increment(stats, "written")
            if target.name == "train":
                stats = increment(stats, "train_written")
            elif target.name == "validation":
                stats = increment(stats, "validation_written")

            if limit is not None and stats.written >= limit:
                break

    return stats


def themes_match(
    row_themes: set[str],
    include_themes: set[str] | None,
    exclude_themes: set[str],
) -> bool:
    if exclude_themes and row_themes.intersection(exclude_themes):
        return False
    if include_themes is None:
        return True
    return bool(row_themes.intersection(include_themes))


def tactical_puzzle_from_lichess_row(row: dict[str, str]) -> TacticalPuzzle | None:
    fen = row.get("FEN")
    moves = (row.get("Moves") or "").split()
    if fen is None or len(moves) < 2:
        return None

    try:
        board = chess.Board(fen)
        first_move = chess.Move.from_uci(moves[0])
    except ValueError:
        return None

    if first_move not in board.legal_moves:
        return None
    board.push(first_move)

    puzzle = TacticalPuzzle(
        initial_fen=board.fen(),
        line_uci=tuple(moves[1:]),
        rating=parse_optional_int(row.get("Rating")),
        themes=tuple(sorted((row.get("Themes") or "").split())),
    )
    try:
        validate_tactical_puzzle(puzzle)
    except ValueError:
        return None
    return puzzle


def agent_move_count_in_range(
    agent_move_count: int,
    *,
    min_agent_moves: int,
    max_agent_moves: int | None,
) -> bool:
    if agent_move_count < min_agent_moves:
        return False
    if max_agent_moves is not None and agent_move_count > max_agent_moves:
        return False
    return True


def write_puzzle(target: OutputHandle, puzzle: TacticalPuzzle) -> None:
    target.handle.write(format_tactical_puzzle_line(puzzle))


def increment(
    stats: TacticalExtractionStats,
    field: str,
) -> TacticalExtractionStats:
    values = stats.__dict__ | {field: getattr(stats, field) + 1}
    return TacticalExtractionStats(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--themes", nargs="*", default=list(DEFAULT_THEMES))
    parser.add_argument("--exclude-themes", nargs="*", default=list(DEFAULT_EXCLUDED_THEMES))
    parser.add_argument("--min-agent-moves", type=int, default=2)
    parser.add_argument("--max-agent-moves", type=int)
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--max-rating", type=int)
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()

    split_outputs = args.train_output is not None or args.validation_output is not None
    output_path = None if split_outputs else args.output or Path("data/tactical_puzzles.txt")
    stats = extract_tactical_puzzles(
        input_path=args.input_csv,
        output_path=output_path,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        limit=args.limit,
        themes=tuple(args.themes) if args.themes else None,
        exclude_themes=tuple(args.exclude_themes),
        min_agent_moves=args.min_agent_moves,
        max_agent_moves=args.max_agent_moves,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        deduplicate=not args.keep_duplicates,
    )
    print_stats(
        output_path=output_path,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        stats=stats,
    )


def print_stats(
    *,
    output_path: Path | None,
    train_output_path: Path | None,
    validation_output_path: Path | None,
    stats: TacticalExtractionStats,
) -> None:
    print("Tactical puzzle extraction")
    if output_path is not None:
        print(f"Output:             {output_path}")
    if train_output_path is not None:
        print(f"Train output:       {train_output_path}")
    if validation_output_path is not None:
        print(f"Validation output:  {validation_output_path}")
    print(f"Rows seen:          {stats.rows_seen}")
    print(f"Theme matches:      {stats.theme_matches}")
    print(f"Rating skips:       {stats.rating_skips}")
    print(f"Length skips:       {stats.length_skips}")
    print(f"Invalid skips:      {stats.invalid_skips}")
    print(f"Duplicate skips:    {stats.duplicate_skips}")
    print(f"Written:            {stats.written}")
    if train_output_path is not None or validation_output_path is not None:
        print(f"Train written:      {stats.train_written}")
        print(f"Validation written: {stats.validation_written}")


if __name__ == "__main__":
    main()
