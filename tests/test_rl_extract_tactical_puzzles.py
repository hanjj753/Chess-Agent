import csv
from pathlib import Path

import chess

from chess_agent.rl.tactical_puzzle_env import load_tactical_puzzles
from chess_agent.utils.extract_tactical_puzzles import (
    extract_tactical_puzzles,
    tactical_puzzle_from_lichess_row,
)


FIELDNAMES = [
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
]


def test_tactical_puzzle_from_lichess_row_applies_first_move_before_exporting() -> None:
    row = tactical_row()

    puzzle = tactical_puzzle_from_lichess_row(row)

    assert puzzle is not None
    board = chess.Board(row["FEN"])
    board.push(chess.Move.from_uci("e2e4"))
    assert puzzle.initial_fen == board.fen()
    assert puzzle.line_uci == ("e7e5", "g1f3", "b8c6")
    assert puzzle.themes == ("fork", "pin")


def test_extract_tactical_puzzles_writes_tsv_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "lichess_puzzles.csv"
    output_path = tmp_path / "tactical.txt"
    write_rows(csv_path, [tactical_row(), mate_in_one_row()])

    stats = extract_tactical_puzzles(
        input_path=csv_path,
        output_path=output_path,
        themes=("fork", "pin"),
        min_agent_moves=2,
    )

    puzzles = load_tactical_puzzles(output_path)
    assert stats.rows_seen == 2
    assert stats.theme_matches == 1
    assert stats.written == 1
    assert len(puzzles) == 1
    assert puzzles[0].line_uci == ("e7e5", "g1f3", "b8c6")


def test_extract_tactical_puzzles_can_write_train_validation_split(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "lichess_puzzles.csv"
    train_path = tmp_path / "tactical_train.txt"
    validation_path = tmp_path / "tactical_valid.txt"
    write_rows(
        csv_path,
        [
            tactical_row(puzzle_id="tactic1"),
            tactical_row(
                puzzle_id="tactic2",
                first_move="d2d4",
                line="d7d5 c1f4 g8f6",
            ),
        ],
    )

    stats = extract_tactical_puzzles(
        input_path=csv_path,
        train_output_path=train_path,
        validation_output_path=validation_path,
        validation_fraction=0.5,
        seed=1,
        themes=("fork", "pin"),
        min_agent_moves=2,
        deduplicate=False,
    )

    train_lines = train_path.read_text(encoding="utf-8").splitlines()
    validation_lines = validation_path.read_text(encoding="utf-8").splitlines()
    assert stats.written == 2
    assert stats.train_written == len(train_lines)
    assert stats.validation_written == len(validation_lines)
    assert stats.train_written + stats.validation_written == stats.written


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def tactical_row(
    *,
    puzzle_id: str = "tactic1",
    first_move: str = "e2e4",
    line: str = "e7e5 g1f3 b8c6",
    rating: str = "1200",
) -> dict[str, str]:
    return {
        "PuzzleId": puzzle_id,
        "FEN": chess.STARTING_FEN,
        "Moves": f"{first_move} {line}",
        "Rating": rating,
        "RatingDeviation": "80",
        "Popularity": "90",
        "NbPlays": "1000",
        "Themes": "fork pin",
        "GameUrl": "https://lichess.org/example#1",
        "OpeningTags": "",
    }


def mate_in_one_row() -> dict[str, str]:
    row = tactical_row()
    return row | {"PuzzleId": "mate1", "Themes": "mateIn1 short"}
