from pathlib import Path

import chess

from chess_agent.rl.tactical_puzzle_env import TacticalPuzzle, format_tactical_puzzle_line
from chess_agent.rl.train_tactical_supervised import (
    TacticalSupervisedTrainingConfig,
    samples_from_puzzles,
    train_tactical_supervised_policy,
)


def test_samples_from_puzzles_collects_agent_turns_only() -> None:
    samples = samples_from_puzzles([make_tactical_puzzle()])

    assert [sample.target_uci for sample in samples] == ["e7e5", "b8c6"]
    assert all(sample.fen for sample in samples)


def test_tactical_supervised_training_runs_and_saves_policy(tmp_path: Path) -> None:
    puzzles_path = write_tactical_file(tmp_path, "train.txt", [make_tactical_puzzle()])
    validation_path = write_tactical_file(tmp_path, "valid.txt", [make_second_tactical_puzzle()])
    save_path = tmp_path / "tactical_policy.pt"
    best_path = tmp_path / "tactical_best.pt"

    _, result = train_tactical_supervised_policy(
        TacticalSupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            validation_file=validation_path,
            epochs=1,
            batch_size=2,
            hidden_size=16,
            log_every=0,
            save_path=save_path,
            best_checkpoint_path=best_path,
        )
    )

    assert save_path.exists()
    assert best_path.exists()
    assert result.best_epoch == 1
    assert 0 <= result.train_accuracy.accuracy <= 1
    assert 0 <= result.validation_puzzle_success_rate <= 1


def test_tactical_supervised_training_can_resume_from_checkpoint(tmp_path: Path) -> None:
    puzzles_path = write_tactical_file(tmp_path, "train.txt", [make_tactical_puzzle()])
    validation_path = write_tactical_file(tmp_path, "valid.txt", [make_second_tactical_puzzle()])
    checkpoint_path = tmp_path / "tactical_checkpoint.pt"

    train_tactical_supervised_policy(
        TacticalSupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            validation_file=validation_path,
            epochs=1,
            batch_size=2,
            hidden_size=16,
            log_every=0,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
        )
    )
    _, result = train_tactical_supervised_policy(
        TacticalSupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            validation_file=validation_path,
            epochs=2,
            batch_size=2,
            hidden_size=16,
            log_every=0,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            resume_from=checkpoint_path,
        )
    )

    assert checkpoint_path.exists()
    assert result.train_accuracy.total == 2


def write_tactical_file(
    tmp_path: Path,
    filename: str,
    puzzles: list[TacticalPuzzle],
) -> Path:
    path = tmp_path / filename
    path.write_text(
        "".join(format_tactical_puzzle_line(puzzle) for puzzle in puzzles),
        encoding="utf-8",
    )
    return path


def make_tactical_puzzle() -> TacticalPuzzle:
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    return TacticalPuzzle(
        initial_fen=board.fen(),
        line_uci=("e7e5", "g1f3", "b8c6"),
        rating=1200,
        themes=("fork", "pin"),
    )


def make_second_tactical_puzzle() -> TacticalPuzzle:
    board = chess.Board()
    board.push(chess.Move.from_uci("d2d4"))
    return TacticalPuzzle(
        initial_fen=board.fen(),
        line_uci=("d7d5", "c1f4", "g8f6"),
        rating=1300,
        themes=("skewer",),
    )
