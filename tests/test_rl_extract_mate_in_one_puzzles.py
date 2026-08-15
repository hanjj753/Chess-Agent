import csv
from pathlib import Path

import chess

from chess_agent.utils.extract_mate_in_one_puzzles import (
    extract_mate_in_one_fens,
    puzzle_from_lichess_row,
)
from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv, load_puzzle_fens


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


def test_puzzle_from_lichess_row_applies_first_move_before_exporting() -> None:
    row = mate_in_one_row()

    puzzle = puzzle_from_lichess_row(row)

    assert puzzle is not None
    assert puzzle.solution_uci == "g6g7"
    board = chess.Board(row["FEN"])
    board.push(chess.Move.from_uci("a8b6"))
    assert puzzle.fen == board.fen()


def test_extract_mate_in_one_fens_writes_fen_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "lichess_puzzles.csv"
    output_path = tmp_path / "mate_in_one_fens.txt"
    write_rows(csv_path, [mate_in_one_row(), non_mate_theme_row()])

    stats = extract_mate_in_one_fens(
        input_path=csv_path,
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert stats.rows_seen == 2
    assert stats.theme_matches == 1
    assert stats.written == 1
    assert len(lines) == 1
    assert ChessMateInOneEnv(puzzles_file=output_path).puzzles == (lines[0],)


def test_extract_mate_in_one_fens_can_include_solution_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "lichess_puzzles.csv"
    output_path = tmp_path / "mate_in_one_fens.txt"
    write_rows(csv_path, [mate_in_one_row()])

    extract_mate_in_one_fens(
        input_path=csv_path,
        output_path=output_path,
        include_solution=True,
    )

    line = output_path.read_text(encoding="utf-8").strip()
    fen, solution, rating = line.split("\t")
    assert solution == "g6g7"
    assert rating == "1200"
    assert load_puzzle_fens(output_path) == (fen,)


def test_extract_mate_in_one_fens_can_write_train_validation_split(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "lichess_puzzles.csv"
    train_path = tmp_path / "mate_in_one_train.txt"
    validation_path = tmp_path / "mate_in_one_valid.txt"
    write_rows(
        csv_path,
        [
            mate_in_one_row(puzzle_id="mate1", opponent_move="a8b6"),
            mate_in_one_row(puzzle_id="mate2", opponent_move="a8c7"),
            mate_in_one_row(puzzle_id="mate3", opponent_move="a8b6"),
        ],
    )

    stats = extract_mate_in_one_fens(
        input_path=csv_path,
        train_output_path=train_path,
        validation_output_path=validation_path,
        validation_fraction=0.5,
        seed=1,
        include_solution=True,
        deduplicate=False,
    )

    train_lines = train_path.read_text(encoding="utf-8").splitlines()
    validation_lines = validation_path.read_text(encoding="utf-8").splitlines()
    assert stats.written == 3
    assert stats.train_written == len(train_lines)
    assert stats.validation_written == len(validation_lines)
    assert stats.train_written + stats.validation_written == stats.written
    assert train_lines
    assert validation_lines
    assert all("\tg6g7\t1200" in line for line in train_lines + validation_lines)


def test_extract_mate_in_one_fens_filters_rating(tmp_path: Path) -> None:
    csv_path = tmp_path / "lichess_puzzles.csv"
    output_path = tmp_path / "mate_in_one_fens.txt"
    write_rows(csv_path, [mate_in_one_row(rating="1200")])

    stats = extract_mate_in_one_fens(
        input_path=csv_path,
        output_path=output_path,
        min_rating=1500,
    )

    assert stats.rating_skips == 1
    assert stats.written == 0
    assert output_path.read_text(encoding="utf-8") == ""


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def mate_in_one_row(
    *,
    puzzle_id: str = "mate1",
    opponent_move: str = "a8b6",
    rating: str = "1200",
) -> dict[str, str]:
    return {
        "PuzzleId": puzzle_id,
        "FEN": "n6k/8/5KQ1/8/8/8/8/8 b - - 0 1",
        "Moves": f"{opponent_move} g6g7",
        "Rating": rating,
        "RatingDeviation": "80",
        "Popularity": "90",
        "NbPlays": "1000",
        "Themes": "mateIn1 short",
        "GameUrl": "https://lichess.org/example#1",
        "OpeningTags": "",
    }


def non_mate_theme_row() -> dict[str, str]:
    row = mate_in_one_row()
    return row | {"PuzzleId": "not-mate1", "Themes": "fork middlegame"}
