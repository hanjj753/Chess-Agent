from pathlib import Path

import chess
from torch.utils.data import WeightedRandomSampler

from chess_agent.rl.tactical_puzzle_env import TacticalPuzzle, format_tactical_puzzle_line
from chess_agent.rl.train_tactical_supervised import (
    TARGET_WEAK_THEME_WEIGHTS,
    TacticalSampleDataset,
    TacticalSupervisedTrainingConfig,
    TacticalTrainingSample,
    make_train_loader,
    resolve_cli_theme_weights,
    samples_from_puzzles,
    training_sample_weight,
    train_tactical_supervised_policy,
)


def test_samples_from_puzzles_collects_agent_turns_only() -> None:
    samples = samples_from_puzzles([make_tactical_puzzle()])

    assert [sample.target_uci for sample in samples] == ["e7e5", "b8c6"]
    assert all(sample.fen for sample in samples)


def test_theme_sample_weight_uses_largest_matching_weight() -> None:
    sample = TacticalTrainingSample(
        fen=chess.STARTING_FEN,
        target_uci="e2e4",
        target_action=0,
        puzzle_index=0,
        line_index=0,
        themes=("quietMove", "trappedPiece"),
    )

    weight = training_sample_weight(
        sample,
        {"quietMove": 3.0, "trappedPiece": 2.5},
    )

    assert weight == 3.0


def test_weighted_train_loader_preserves_epoch_sample_count() -> None:
    samples = samples_from_puzzles([make_tactical_puzzle()])
    dataset = TacticalSampleDataset(samples)

    loader = make_train_loader(
        dataset=dataset,
        batch_size=2,
        seed=0,
        theme_weights={"fork": 2.0},
    )

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.num_samples == len(dataset)


def test_target_theme_profile_can_be_overridden() -> None:
    weights = dict(
        resolve_cli_theme_weights(
            target_weak_themes=True,
            custom_weights=(("quietMove", 4.0), ("fork", 1.5)),
        )
    )

    assert dict(TARGET_WEAK_THEME_WEIGHTS)["quietMove"] == 3.0
    assert weights["quietMove"] == 4.0
    assert weights["fork"] == 1.5


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
            architecture="mlp",
            hidden_size=16,
            dropout=0.0,
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


def test_tactical_training_can_write_experiment_records(tmp_path: Path) -> None:
    puzzles_path = write_tactical_file(tmp_path, "train.txt", [make_tactical_puzzle()])
    validation_path = write_tactical_file(
        tmp_path,
        "valid.txt",
        [make_second_tactical_puzzle()],
    )
    experiment_dir = tmp_path / "experiments"

    train_tactical_supervised_policy(
        TacticalSupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            validation_file=validation_path,
            epochs=1,
            batch_size=2,
            architecture="mlp",
            hidden_size=16,
            dropout=0.0,
            log_every=0,
            experiment_dir=experiment_dir,
            experiment_name="test tactical",
        )
    )

    run_dirs = list(experiment_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "config.json").exists()
    assert "validation_accuracy" in (run_dirs[0] / "metrics.csv").read_text(
        encoding="utf-8"
    )
    assert (run_dirs[0] / "summary.json").exists()


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
            architecture="mlp",
            hidden_size=16,
            dropout=0.0,
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
            architecture="mlp",
            hidden_size=16,
            dropout=0.0,
            log_every=0,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            resume_from=checkpoint_path,
        )
    )

    assert checkpoint_path.exists()
    assert result.train_accuracy.total == 2


def test_tactical_supervised_training_stops_after_patience(tmp_path: Path) -> None:
    puzzles_path = write_tactical_file(tmp_path, "train.txt", [make_tactical_puzzle()])
    validation_path = write_tactical_file(
        tmp_path,
        "valid.txt",
        [make_second_tactical_puzzle()],
    )

    _, result = train_tactical_supervised_policy(
        TacticalSupervisedTrainingConfig(
            puzzles_file=puzzles_path,
            validation_file=validation_path,
            epochs=5,
            batch_size=2,
            learning_rate=0.0,
            architecture="mlp",
            hidden_size=16,
            dropout=0.0,
            early_stopping_patience=1,
            log_every=0,
        )
    )

    assert result.stopped_early
    assert result.completed_epochs == 2
    assert result.best_epoch == 1


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
