import argparse
import contextlib
import csv
from dataclasses import dataclass
import gzip
import io
from pathlib import Path
import random
from typing import Iterator, TextIO

import chess


@dataclass(frozen=True)
class ExtractedPuzzle:
    fen: str
    solution_uci: str
    rating: int | None


@dataclass(frozen=True)
class ExtractionStats:
    rows_seen: int = 0
    theme_matches: int = 0
    rating_skips: int = 0
    invalid_skips: int = 0
    duplicate_skips: int = 0
    written: int = 0
    train_written: int = 0
    validation_written: int = 0


def extract_mate_in_one_fens(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    train_output_path: str | Path | None = None,
    validation_output_path: str | Path | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
    limit: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    include_solution: bool = False,
    deduplicate: bool = True,
) -> ExtractionStats:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
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

    stats = ExtractionStats()
    seen_fens: set[str] = set()
    rng = random.Random(seed)

    with (
        open_puzzle_csv(input_path) as handle,
        contextlib.ExitStack() as stack,
    ):
        writer = open_output_writers(
            stack=stack,
            output_path=output_path,
            train_output_path=train_output_path,
            validation_output_path=validation_output_path,
        )
        reader = csv.DictReader(handle)
        for row in reader:
            stats = increment(stats, "rows_seen")
            if not is_mate_in_one_row(row):
                continue

            stats = increment(stats, "theme_matches")
            rating = parse_optional_int(row.get("Rating"))
            if rating_out_of_range(rating, min_rating, max_rating):
                stats = increment(stats, "rating_skips")
                continue

            puzzle = puzzle_from_lichess_row(row)
            if puzzle is None:
                stats = increment(stats, "invalid_skips")
                continue

            if deduplicate and puzzle.fen in seen_fens:
                stats = increment(stats, "duplicate_skips")
                continue

            seen_fens.add(puzzle.fen)
            target = select_writer(
                writer,
                rng=rng,
                validation_fraction=validation_fraction,
            )
            target.handle.write(format_puzzle_line(puzzle, include_solution=include_solution))
            stats = increment(stats, "written")
            if target.name == "train":
                stats = increment(stats, "train_written")
            elif target.name == "validation":
                stats = increment(stats, "validation_written")

            if limit is not None and stats.written >= limit:
                break

    return stats


@dataclass(frozen=True)
class OutputHandle:
    name: str
    handle: TextIO


@dataclass(frozen=True)
class OutputWriters:
    single: OutputHandle | None = None
    train: OutputHandle | None = None
    validation: OutputHandle | None = None


def open_output_writers(
    *,
    stack: contextlib.ExitStack,
    output_path: str | Path | None,
    train_output_path: str | Path | None,
    validation_output_path: str | Path | None,
) -> OutputWriters:
    if output_path is not None:
        return OutputWriters(
            single=open_output_handle(stack, "single", output_path),
        )

    if train_output_path is None or validation_output_path is None:
        raise ValueError("split output paths are incomplete")

    return OutputWriters(
        train=open_output_handle(stack, "train", train_output_path),
        validation=open_output_handle(stack, "validation", validation_output_path),
    )


def open_output_handle(
    stack: contextlib.ExitStack,
    name: str,
    path: str | Path,
) -> OutputHandle:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return OutputHandle(
        name=name,
        handle=stack.enter_context(output.open("w", encoding="utf-8")),
    )


def select_writer(
    writers: OutputWriters,
    *,
    rng: random.Random,
    validation_fraction: float,
) -> OutputHandle:
    if writers.single is not None:
        return writers.single

    if writers.train is None or writers.validation is None:
        raise ValueError("split writers are incomplete")

    if rng.random() < validation_fraction:
        return writers.validation
    return writers.train


@contextlib.contextmanager
def open_puzzle_csv(path: str | Path) -> Iterator[TextIO]:
    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
        return

    if input_path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading .zst files requires the zstandard package. "
                "Install it with: .\\.venv\\Scripts\\python -m pip install zstandard"
            ) from exc

        with input_path.open("rb") as compressed:
            reader = zstd.ZstdDecompressor().stream_reader(compressed)
            text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
            try:
                yield text
            finally:
                text.close()
        return

    with input_path.open(encoding="utf-8", newline="") as handle:
        yield handle


def is_mate_in_one_row(row: dict[str, str]) -> bool:
    return "mateIn1" in set((row.get("Themes") or "").split())


def rating_out_of_range(
    rating: int | None,
    min_rating: int | None,
    max_rating: int | None,
) -> bool:
    if rating is None:
        return False
    if min_rating is not None and rating < min_rating:
        return True
    if max_rating is not None and rating > max_rating:
        return True
    return False


def puzzle_from_lichess_row(row: dict[str, str]) -> ExtractedPuzzle | None:
    fen = row.get("FEN")
    moves = (row.get("Moves") or "").split()
    if fen is None or len(moves) < 2:
        return None

    try:
        board = chess.Board(fen)
        opponent_move = chess.Move.from_uci(moves[0])
        solution_move = chess.Move.from_uci(moves[1])
    except ValueError:
        return None

    if opponent_move not in board.legal_moves:
        return None

    board.push(opponent_move)
    if solution_move not in board.legal_moves:
        return None

    after_solution = board.copy(stack=False)
    after_solution.push(solution_move)
    if not after_solution.is_checkmate():
        return None

    return ExtractedPuzzle(
        fen=board.fen(),
        solution_uci=solution_move.uci(),
        rating=parse_optional_int(row.get("Rating")),
    )


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def format_puzzle_line(puzzle: ExtractedPuzzle, *, include_solution: bool) -> str:
    if not include_solution:
        return f"{puzzle.fen}\n"

    rating = "" if puzzle.rating is None else str(puzzle.rating)
    return f"{puzzle.fen}\t{puzzle.solution_uci}\t{rating}\n"


def increment(stats: ExtractionStats, field: str) -> ExtractionStats:
    values = stats.__dict__ | {field: getattr(stats, field) + 1}
    return ExtractionStats(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--max-rating", type=int)
    parser.add_argument("--include-solution", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()
    split_outputs = args.train_output is not None or args.validation_output is not None
    output_path = None if split_outputs else args.output or Path("data/mate_in_one_fens.txt")

    stats = extract_mate_in_one_fens(
        input_path=args.input_csv,
        output_path=output_path,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        limit=args.limit,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        include_solution=args.include_solution,
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
    stats: ExtractionStats,
) -> None:
    print("Mate-in-one extraction")
    if output_path is not None:
        print(f"Output:           {output_path}")
    if train_output_path is not None:
        print(f"Train output:     {train_output_path}")
    if validation_output_path is not None:
        print(f"Validation output: {validation_output_path}")
    print(f"Rows seen:        {stats.rows_seen}")
    print(f"Theme matches:    {stats.theme_matches}")
    print(f"Rating skips:     {stats.rating_skips}")
    print(f"Invalid skips:    {stats.invalid_skips}")
    print(f"Duplicate skips:  {stats.duplicate_skips}")
    print(f"Written:          {stats.written}")
    if train_output_path is not None or validation_output_path is not None:
        print(f"Train written:    {stats.train_written}")
        print(f"Validation written: {stats.validation_written}")


if __name__ == "__main__":
    main()
