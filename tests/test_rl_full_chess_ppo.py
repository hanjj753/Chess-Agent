import csv
from pathlib import Path

import pytest
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import is_masking_supported

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


def test_full_chess_ppo_smoke_training_saves_logs_and_models(tmp_path: Path) -> None:
    config = smoke_config(
        total_timesteps=4,
        evaluation_every=4,
        checkpoint_every=4,
        experiment_dir=tmp_path / "experiments",
        save_path=tmp_path / "final.zip",
        best_model_path=tmp_path / "best.zip",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    model, result = train_full_chess_ppo(config)

    assert isinstance(model, MaskablePPO)
    assert result.completed_timesteps == 4
    assert result.final_model_path.exists()
    assert result.best_model_path.exists()
    assert model.target_kl == config.target_kl
    assert result.experiment_run_dir is not None
    assert (result.experiment_run_dir / "summary.json").exists()
    metrics = (result.experiment_run_dir / "metrics.csv").read_text(encoding="utf-8")
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
