from pathlib import Path

import chess

from chess_agent.rl.actions import move_to_action
from chess_agent.rl.train_mate_in_one import TrainingConfig, train_policy_gradient
from chess_agent.rl.train_mate_in_one_supervised import (
    SupervisedTrainingConfig,
    load_labeled_puzzles,
    train_supervised_policy,
)


def test_load_labeled_puzzles_reads_solution_metadata(tmp_path: Path) -> None:
    path = tmp_path / "puzzles.txt"
    path.write_text(
        "7k/8/5KQ1/8/8/8/8/8 w - - 0 1\tg6g7\t1200\n",
        encoding="utf-8",
    )

    puzzles = load_labeled_puzzles(path)

    assert len(puzzles) == 1
    assert puzzles[0].solution_uci == "g6g7"
    assert puzzles[0].target_action == move_to_action(chess.Move.from_uci("g6g7"))
    assert puzzles[0].rating == 1200


def test_load_labeled_puzzles_derives_solution_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "puzzles.txt"
    path.write_text("7k/8/5KQ1/8/8/8/8/8 w - - 0 1\n", encoding="utf-8")

    puzzles = load_labeled_puzzles(path)

    assert puzzles[0].solution_uci == "g6g7"


def test_supervised_training_runs_and_saves_policy(tmp_path: Path) -> None:
    puzzles_path = write_tiny_puzzle_file(tmp_path)
    save_path = tmp_path / "supervised_policy.pt"
    best_path = tmp_path / "supervised_best.pt"

    _, result = train_supervised_policy(
        SupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            epochs=1,
            batch_size=2,
            hidden_size=16,
            save_path=save_path,
            best_checkpoint_path=best_path,
        )
    )

    assert save_path.exists()
    assert best_path.exists()
    assert result.best_epoch == 1
    assert 0 <= result.train_accuracy.accuracy <= 1
    assert 0 <= result.validation_mate_success_rate <= 1


def test_supervised_training_can_use_separate_validation_file(tmp_path: Path) -> None:
    puzzles_path = write_tiny_puzzle_file(tmp_path)
    validation_path = tmp_path / "validation.txt"
    validation_path.write_text(
        "7k/8/5KQ1/8/8/8/8/8 w - - 0 1\tg6g7\t1200\n",
        encoding="utf-8",
    )

    _, result = train_supervised_policy(
        SupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            validation_file=validation_path,
            epochs=0,
            batch_size=2,
            hidden_size=16,
            log_every=0,
        )
    )

    assert result.train_accuracy.total == 4
    assert result.validation_accuracy.total == 1


def test_supervised_training_can_resume_from_checkpoint(tmp_path: Path) -> None:
    puzzles_path = write_tiny_puzzle_file(tmp_path)
    checkpoint_path = tmp_path / "supervised_checkpoint.pt"

    train_supervised_policy(
        SupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            epochs=1,
            batch_size=2,
            hidden_size=16,
            log_every=0,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
        )
    )
    _, result = train_supervised_policy(
        SupervisedTrainingConfig(
            puzzles_file=puzzles_path,
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
    assert result.train_accuracy.total > 0


def test_policy_gradient_can_start_from_supervised_policy(tmp_path: Path) -> None:
    puzzles_path = write_tiny_puzzle_file(tmp_path)
    save_path = tmp_path / "supervised_policy.pt"

    train_supervised_policy(
        SupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            epochs=1,
            batch_size=2,
            hidden_size=16,
            save_path=save_path,
        )
    )
    _, result = train_policy_gradient(
        config=TrainingConfig(
            episodes=1,
            log_every=0,
            puzzles_file=puzzles_path,
            pretrained_path=save_path,
        )
    )

    assert result.episodes == 1


def test_policy_gradient_can_use_separate_evaluation_file(tmp_path: Path) -> None:
    puzzles_path = write_tiny_puzzle_file(tmp_path)
    evaluation_path = tmp_path / "evaluation.txt"
    evaluation_path.write_text(
        "\n".join(
            [
                "7k/8/5KQ1/8/8/8/8/8 w - - 0 1\tg6g7\t1200",
                "8/8/8/8/8/5kq1/8/7K b - - 0 1\tg3g2\t1200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, result = train_policy_gradient(
        config=TrainingConfig(
            episodes=1,
            hidden_size=16,
            log_every=0,
            puzzles_file=puzzles_path,
            evaluation_puzzles_file=evaluation_path,
        )
    )

    assert result.final_evaluation.episodes == 2


def test_policy_gradient_can_resume_from_checkpoint(tmp_path: Path) -> None:
    puzzles_path = write_tiny_puzzle_file(tmp_path)
    checkpoint_path = tmp_path / "rl_checkpoint.pt"

    train_policy_gradient(
        config=TrainingConfig(
            episodes=1,
            hidden_size=16,
            log_every=0,
            puzzles_file=puzzles_path,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
        )
    )
    _, result = train_policy_gradient(
        config=TrainingConfig(
            episodes=2,
            log_every=0,
            puzzles_file=puzzles_path,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            resume_from=checkpoint_path,
        )
    )

    assert checkpoint_path.exists()
    assert result.episodes == 2


def write_tiny_puzzle_file(tmp_path: Path) -> Path:
    path = tmp_path / "puzzles.txt"
    path.write_text(
        "\n".join(
            [
                "7k/8/5KQ1/8/8/8/8/8 w - - 0 1\tg6g7\t1200",
                "8/8/8/8/8/5kq1/8/7K b - - 0 1\tg3g2\t1200",
                "6k1/8/6K1/8/8/8/8/R7 w - - 0 1\ta1a8\t1200",
                "r7/8/8/8/8/6k1/8/6K1 b - - 0 1\ta8a1\t1200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
