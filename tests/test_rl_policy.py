from pathlib import Path

import torch

from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv
from chess_agent.rl.policy import (
    MASKED_LOGIT,
    ConvolutionalPolicy,
    MateInOnePolicy,
    apply_action_mask,
)
from chess_agent.rl.train_mate_in_one import (
    TrainingConfig,
    evaluate_policy,
    greedy_action,
    load_policy,
    save_policy,
    train_policy_gradient,
)


def test_policy_outputs_one_logit_per_action() -> None:
    env = ChessMateInOneEnv()
    observation, _ = env.reset(options={"puzzle_index": 0})
    policy = MateInOnePolicy(hidden_size=16)
    board = torch.as_tensor(observation["board"]).unsqueeze(0)

    logits = policy(board)

    assert logits.shape == (1, env.action_space.n)


def test_convolutional_policy_outputs_one_logit_per_action() -> None:
    env = ChessMateInOneEnv()
    observation, _ = env.reset(options={"puzzle_index": 0})
    policy = ConvolutionalPolicy(hidden_size=8, dropout=0.1, residual_blocks=1)
    board = torch.as_tensor(observation["board"]).unsqueeze(0)

    logits = policy(board)

    assert logits.shape == (1, env.action_space.n)


def test_convolutional_policy_save_and_load_preserves_architecture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cnn_policy.pt"
    policy = ConvolutionalPolicy(hidden_size=8, dropout=0.2, residual_blocks=1)

    save_policy(policy, path)
    loaded = load_policy(path)

    assert isinstance(loaded, ConvolutionalPolicy)
    assert loaded.hidden_size == 8
    assert loaded.dropout_rate == 0.2
    assert loaded.residual_blocks == 1


def test_load_policy_supports_legacy_mlp_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "legacy_mlp.pt"
    policy = MateInOnePolicy(hidden_size=8)
    torch.save(
        {
            "hidden_size": policy.hidden_size,
            "state_dict": policy.state_dict(),
        },
        path,
    )

    loaded = load_policy(path)

    assert isinstance(loaded, MateInOnePolicy)
    assert loaded.hidden_size == 8


def test_apply_action_mask_blocks_illegal_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])

    masked = apply_action_mask(logits, mask)

    assert masked[0, 0] == 1.0
    assert masked[0, 1] == MASKED_LOGIT
    assert masked[0, 2] == 3.0


def test_greedy_action_respects_action_mask() -> None:
    env = ChessMateInOneEnv()
    observation, _ = env.reset(options={"puzzle_index": 0})
    policy = MateInOnePolicy(hidden_size=16)

    action = greedy_action(policy, observation, torch.device("cpu"))

    assert observation["action_mask"][action] == 1


def test_policy_gradient_training_runs() -> None:
    env = ChessMateInOneEnv()
    _, result = train_policy_gradient(
        config=TrainingConfig(
            episodes=2,
            hidden_size=16,
            log_every=0,
        ),
        env=env,
    )

    assert result.episodes == 2
    assert 0 <= result.successes <= 2
    assert 0 <= result.final_evaluation.success_rate <= 1


def test_policy_gradient_can_limit_evaluation_episodes() -> None:
    env = ChessMateInOneEnv()
    _, result = train_policy_gradient(
        config=TrainingConfig(
            episodes=1,
            hidden_size=16,
            log_every=0,
            evaluation_episodes=1,
        ),
        env=env,
    )

    assert result.final_evaluation.episodes == 1


def test_evaluate_policy_counts_episodes() -> None:
    result = evaluate_policy(
        policy=MateInOnePolicy(hidden_size=16),
        env=ChessMateInOneEnv(),
        episodes=4,
    )

    assert result.episodes == 4
    assert 0 <= result.successes <= 4


def test_evaluate_policy_preserves_policy_device_by_default() -> None:
    policy = MateInOnePolicy(hidden_size=16)
    before_device = next(policy.parameters()).device

    evaluate_policy(
        policy=policy,
        env=ChessMateInOneEnv(),
        episodes=1,
    )

    assert next(policy.parameters()).device == before_device
