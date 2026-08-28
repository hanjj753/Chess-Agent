import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import is_masking_supported

from chess_agent.rl.actions import ACTION_SIZE
from chess_agent.rl.observations import history_observation_shape
from chess_agent.rl.policy_value import PolicyValueNetwork
from chess_agent.rl.ppo_policy import (
    ChessMaskableActorCriticPolicy,
    transfer_policy_value_to_ppo,
)
from chess_agent.rl.train_full_chess_ppo import (
    FullChessPPOConfig,
    TrackedMaskablePPO,
    evaluate_full_chess_ppo,
    make_vector_env,
    train_full_chess_ppo,
    validate_config,
)


def test_full_chess_vector_env_supports_action_masking() -> None:
    env = make_vector_env(smoke_config())
    try:
        assert env.observation_space.shape == history_observation_shape(1)
        assert is_masking_supported(env)
    finally:
        env.close()


def test_policy_value_transfer_preserves_actor_and_critic_outputs() -> None:
    config = smoke_config()
    env = make_vector_env(config)
    try:
        source = PolicyValueNetwork(
            input_channels=history_observation_shape(1)[0],
            hidden_size=8,
            dropout=0.0,
            residual_blocks=1,
        )
        model = TrackedMaskablePPO(
            ChessMaskableActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={
                "hidden_size": 8,
                "dropout": 0.0,
                "residual_blocks": 1,
            },
            device="cpu",
        )
        target = model.policy
        assert isinstance(target, ChessMaskableActorCriticPolicy)
        transfer_policy_value_to_ppo(source=source, target=target)
        source.eval()
        target.eval()

        observation = torch.zeros(1, history_observation_shape(1)[0], 8, 8)
        source_logits, source_value = source(observation)
        features = target.extract_features(observation)
        actor_latent, critic_latent = target.mlp_extractor(features)
        target_logits = target.action_net(actor_latent)
        target_value = target.value_net(critic_latent).flatten()

        torch.testing.assert_close(target_logits, source_logits)
        torch.testing.assert_close(torch.tanh(target_value), source_value)
    finally:
        env.close()


def test_ppo_policy_reproduces_log_probabilities_in_training_mode() -> None:
    config = smoke_config()
    env = make_vector_env(config)
    try:
        model = TrackedMaskablePPO(
            ChessMaskableActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={
                "hidden_size": 8,
                "dropout": 0.5,
                "residual_blocks": 1,
            },
            device="cpu",
        )
        policy = model.policy
        observations = torch.randn(
            2,
            history_observation_shape(1)[0],
            8,
            8,
        )
        action_masks = torch.ones(2, ACTION_SIZE, dtype=torch.bool)

        policy.set_training_mode(False)
        with torch.no_grad():
            distribution = policy.get_distribution(
                observations,
                action_masks=action_masks,
            )
            actions = distribution.get_actions(deterministic=True)
            rollout_log_prob = distribution.log_prob(actions)
        batch_norm_state = {
            name: module.running_mean.clone()
            for name, module in policy.named_modules()
            if isinstance(module, nn.BatchNorm2d)
        }

        policy.set_training_mode(True)
        with torch.no_grad():
            _, update_log_prob, _ = policy.evaluate_actions(
                observations,
                actions,
                action_masks=action_masks,
            )

        assert policy.training
        assert all(
            not module.training
            for module in policy.modules()
            if isinstance(module, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d))
        )
        torch.testing.assert_close(update_log_prob, rollout_log_prob)
        for name, module in policy.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                torch.testing.assert_close(
                    module.running_mean,
                    batch_norm_state[name],
                )
    finally:
        env.close()


def test_full_chess_ppo_smoke_training_saves_logs_and_models(tmp_path: Path) -> None:
    config = smoke_config(
        total_timesteps=4,
        n_epochs=2,
        evaluation_every=4,
        checkpoint_every=4,
        experiment_dir=tmp_path / "experiments",
        save_path=tmp_path / "final.zip",
        initial_model_path=tmp_path / "initial.zip",
        best_model_path=tmp_path / "best.zip",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    model, result = train_full_chess_ppo(config)

    assert isinstance(model, MaskablePPO)
    assert result.completed_timesteps == 4
    assert result.initial_model_path.exists()
    assert result.final_model_path.exists()
    assert result.best_model_path.exists()
    assert model.target_kl == config.target_kl
    assert model._n_updates == 4
    assert model.policy.optimizer.state
    assert result.experiment_run_dir is not None
    assert (result.experiment_run_dir / "summary.json").exists()
    metrics_path = result.experiment_run_dir / "metrics.csv"
    metrics = metrics_path.read_text(encoding="utf-8")
    games = (result.experiment_run_dir / "games.csv").read_text(encoding="utf-8")
    assert "policy_loss" in metrics
    assert "score_rate" in metrics
    assert "final_evaluation" in games
    with (result.experiment_run_dir / "games.csv").open(
        encoding="utf-8",
        newline="",
    ) as source:
        game_rows = list(csv.DictReader(source))
    assert any(
        row["phase"] == "evaluation" and row["step"] == "0"
        for row in game_rows
    )
    with metrics_path.open(encoding="utf-8", newline="") as source:
        metric_rows = list(csv.DictReader(source))
    update_kl = [
        float(row["value"])
        for row in metric_rows
        if row["metric"] == "approx_kl"
    ]
    clip_fractions = [
        float(row["value"])
        for row in metric_rows
        if row["metric"] == "clip_fraction"
    ]
    rollout_metric_names = {
        row["metric"]
        for row in metric_rows
        if row["phase"] == "rollout"
    }
    assert update_kl
    assert max(update_kl) < 1.5 * config.target_kl
    assert clip_fractions and max(clip_fractions) < 0.5
    assert {
        "transitions",
        "completed_games",
        "decisive_games",
        "reward_signal_rate",
        "return_std",
        "value_prediction_std",
        "advantage_std",
    } <= rollout_metric_names

    events = [
        json.loads(line)
        for line in (result.experiment_run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    best_steps = [
        event["step"]
        for event in events
        if event.get("event") == "checkpoint" and event.get("is_best")
    ]
    assert best_steps == [0]


def test_full_chess_ppo_evaluation_uses_only_legal_actions() -> None:
    config = smoke_config()
    env = make_vector_env(config)
    try:
        model = TrackedMaskablePPO(
            ChessMaskableActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={
                "hidden_size": 8,
                "dropout": 0.0,
                "residual_blocks": 1,
            },
            device="cpu",
        )
        result = evaluate_full_chess_ppo(
            model=model,
            episodes=2,
            history_length=1,
            max_plies=2,
            opponent="random",
            opponent_depth=1,
            opponent_time_limit=None,
            deterministic=True,
            seed=0,
        )
    finally:
        env.close()

    assert result.episodes == 2
    assert result.illegal_actions == 0


def test_full_chess_ppo_rejects_invalid_target_kl() -> None:
    with pytest.raises(ValueError, match="target_kl"):
        validate_config(smoke_config(target_kl=0.0))


def test_full_chess_ppo_rejects_dropout() -> None:
    with pytest.raises(ValueError, match="dropout=0"):
        validate_config(smoke_config(dropout=0.1))


def smoke_config(**overrides: object) -> FullChessPPOConfig:
    values = {
        "total_timesteps": 0,
        "n_envs": 1,
        "n_steps": 2,
        "batch_size": 2,
        "n_epochs": 1,
        "history_length": 1,
        "max_plies": 2,
        "hidden_size": 8,
        "dropout": 0.0,
        "residual_blocks": 1,
        "evaluation_every": 0,
        "evaluation_games": 1,
        "checkpoint_every": 0,
        "device": "cpu",
        "experiment_dir": None,
    }
    values.update(overrides)
    return FullChessPPOConfig(**values)
