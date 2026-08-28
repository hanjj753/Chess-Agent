from pathlib import Path

import numpy as np
import torch

from chess_agent.rl.collect_value_dataset import collect_value_dataset
from chess_agent.rl.evaluate_value_head import (
    build_value_evaluation_report,
    evaluate_value_checkpoint,
)
from chess_agent.rl.policy_value import (
    PolicyValueNetwork,
    load_policy_value,
    save_policy_value,
)
from chess_agent.rl.pretrain_value_head import (
    ValuePretrainingConfig,
    make_sample_weights,
    pretrain_value_head,
)
from chess_agent.rl.report_experiment import generate_experiment_report
from chess_agent.rl.value_dataset import (
    load_value_dataset,
    pack_observations,
    save_value_dataset,
    summarize_value_dataset,
)


def test_value_dataset_pack_save_and_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    observations = rng.integers(0, 2, size=(4, 18, 8, 8), dtype=np.int8)
    path = tmp_path / "values.npz"

    save_value_dataset(
        path,
        packed_observations=pack_observations(observations),
        targets=np.asarray((1.0, 0.5, 0.0, -1.0), dtype=np.float32),
        outcomes=np.asarray((1, 1, 0, -1), dtype=np.int8),
        game_ids=np.asarray((1, 1, 2, 3), dtype=np.int32),
        observation_shape=(18, 8, 8),
        metadata={"split": "train"},
    )

    dataset = load_value_dataset(path)

    assert len(dataset) == 4
    assert dataset.games == 3
    assert dataset.metadata["split"] == "train"
    np.testing.assert_array_equal(dataset.unpack(np.arange(4)), observations)
    summary = summarize_value_dataset(dataset)
    assert (summary.wins, summary.draws, summary.losses) == (1, 1, 1)
    assert summary.positions == 4


def test_collect_value_dataset_splits_whole_games(tmp_path: Path) -> None:
    model_path = tmp_path / "source.pt"
    save_policy_value(
        PolicyValueNetwork(
            input_channels=18,
            hidden_size=8,
            dropout=0.0,
            residual_blocks=1,
        ),
        model_path,
    )

    result = collect_value_dataset(
        model_path=model_path,
        train_output_path=tmp_path / "train.npz",
        validation_output_path=tmp_path / "validation.npz",
        games=4,
        validation_fraction=0.25,
        opponent="random",
        alpha_fraction=0.0,
        opponent_depth=1,
        opponent_time_limit=None,
        max_plies=2,
        gamma=0.995,
        deterministic_policy=True,
        temperature=1.0,
        seed=0,
        device="cpu",
        log_every=0,
    )
    train = load_value_dataset(result.train_path)
    validation = load_value_dataset(result.validation_path)

    assert result.train_games == 3
    assert result.validation_games == 1
    assert len(train) + len(validation) == 4
    assert not set(train.game_ids) & set(validation.game_ids)
    assert train.observation_shape == (18, 8, 8)


def test_value_pretraining_changes_only_value_head(tmp_path: Path) -> None:
    model_path = tmp_path / "source.pt"
    source = PolicyValueNetwork(
        input_channels=18,
        hidden_size=8,
        dropout=0.0,
        residual_blocks=1,
    )
    save_policy_value(source, model_path)
    frozen_before = {
        name: value.clone()
        for name, value in source.state_dict().items()
        if not name.startswith("value_head.")
    }
    value_before = {
        name: value.clone()
        for name, value in source.value_head.state_dict().items()
    }

    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    make_synthetic_dataset(train_path, game_offset=0, games=6)
    make_synthetic_dataset(validation_path, game_offset=100, games=3)

    model, result = pretrain_value_head(
        ValuePretrainingConfig(
            model_path=model_path,
            train_data_path=train_path,
            validation_data_path=validation_path,
            epochs=2,
            batch_size=4,
            learning_rate=1e-2,
            patience=0,
            device="cpu",
            save_path=tmp_path / "final.pt",
            best_model_path=tmp_path / "best.pt",
            experiment_dir=tmp_path / "experiments",
        )
    )

    current_state = model.state_dict()
    for name, expected in frozen_before.items():
        torch.testing.assert_close(current_state[name], expected)
    assert any(
        not torch.equal(model.value_head.state_dict()[name], expected)
        for name, expected in value_before.items()
    )
    assert result.completed_epochs == 2
    assert result.final_model_path.is_file()
    assert result.best_model_path.is_file()
    assert result.experiment_run_dir is not None
    assert (result.experiment_run_dir / "metrics.csv").is_file()
    assert result.train_dataset.games == 6
    assert result.validation_dataset.games == 3

    report = generate_experiment_report(result.experiment_run_dir)
    assert "Value-head supervised pretraining 보고서" in report.summary_path.read_text(
        encoding="utf-8"
    )
    assert "opponent=random, max_plies=100" in report.summary_path.read_text(
        encoding="utf-8"
    )
    assert report.learning_curves_path is not None
    assert report.learning_curves_path.stat().st_size > 1_000
    assert report.game_outcomes_path is None

    loaded = load_policy_value(result.final_model_path)
    assert loaded.input_channels == 18


def test_value_sample_weights_balance_games_and_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "values.npz"
    observations = np.zeros((6, 18, 8, 8), dtype=np.int8)
    save_value_dataset(
        path,
        packed_observations=pack_observations(observations),
        targets=np.asarray((1.0, 0.9, 0.8, 0.0, -1.0, -0.8), dtype=np.float32),
        outcomes=np.asarray((1, 1, 1, 0, -1, -1), dtype=np.int8),
        game_ids=np.asarray((1, 1, 1, 2, 3, 3), dtype=np.int32),
        observation_shape=(18, 8, 8),
    )
    dataset = load_value_dataset(path)

    weights = make_sample_weights(
        dataset,
        balance_games=True,
        balance_outcomes=True,
    )

    class_weights = [float(np.sum(weights[dataset.outcomes == value])) for value in (-1, 0, 1)]
    np.testing.assert_allclose(class_weights, class_weights[0])


def test_evaluate_value_checkpoint_writes_dataset_metrics(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    data_path = tmp_path / "validation.npz"
    save_policy_value(
        PolicyValueNetwork(
            input_channels=18,
            hidden_size=8,
            dropout=0.0,
            residual_blocks=1,
        ),
        model_path,
    )
    make_synthetic_dataset(data_path, game_offset=0, games=3)

    metrics, dataset = evaluate_value_checkpoint(
        model_path=model_path,
        data_path=data_path,
        batch_size=4,
        device="cpu",
    )
    report = build_value_evaluation_report(
        model_path=model_path,
        data_path=data_path,
        metrics=metrics,
        dataset=dataset,
    )

    assert dataset.games == 3
    assert "Opponent:       random" in report
    assert "Max plies:      100" in report
    assert "Explained var:" in report


def make_synthetic_dataset(path: Path, *, game_offset: int, games: int) -> None:
    observations = np.zeros((games * 2, 18, 8, 8), dtype=np.int8)
    game_ids = np.repeat(np.arange(game_offset + 1, game_offset + games + 1), 2)
    outcomes = np.resize(np.asarray((-1, 0, 1), dtype=np.int8), games)
    sample_outcomes = np.repeat(outcomes, 2)
    for index, outcome in enumerate(sample_outcomes):
        observations[index, outcome + 1, 0, 0] = 1
    save_value_dataset(
        path,
        packed_observations=pack_observations(observations),
        targets=sample_outcomes.astype(np.float32),
        outcomes=sample_outcomes,
        game_ids=game_ids.astype(np.int32),
        observation_shape=(18, 8, 8),
        metadata={
            "opponent": "random",
            "max_plies": 100,
            "gamma": 0.995,
        },
    )
