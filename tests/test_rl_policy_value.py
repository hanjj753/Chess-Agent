from pathlib import Path

import chess
import torch

from chess_agent.rl.actions import ACTION_SIZE
from chess_agent.rl.observations import board_to_observation, history_observation_shape
from chess_agent.rl.policy import ConvolutionalPolicy
from chess_agent.rl.policy_value import (
    PolicyValueNetwork,
    create_policy_value_from_policy,
    load_policy_value,
    save_policy_value,
)
from chess_agent.rl.initialize_policy_value import initialize_policy_value
from chess_agent.rl.train_mate_in_one import save_policy


def test_policy_value_network_outputs_logits_and_bounded_value() -> None:
    model = PolicyValueNetwork(
        input_channels=history_observation_shape(4)[0],
        hidden_size=8,
        dropout=0.0,
        residual_blocks=1,
    )
    boards = torch.zeros(2, history_observation_shape(4)[0], 8, 8)

    logits, values = model(boards)

    assert logits.shape == (2, ACTION_SIZE)
    assert values.shape == (2,)
    assert torch.all(values >= -1)
    assert torch.all(values <= 1)


def test_tactical_policy_transfer_preserves_policy_logits() -> None:
    torch.manual_seed(0)
    policy = ConvolutionalPolicy(
        hidden_size=8,
        dropout=0.0,
        residual_blocks=1,
    )
    model = create_policy_value_from_policy(
        policy,
        input_channels=history_observation_shape(2)[0],
    )
    policy.eval()
    model.eval()

    current = torch.as_tensor(board_to_observation(chess.Board())).unsqueeze(0)
    history = torch.randn(1, history_observation_shape(2)[0], 8, 8)
    history[:, : current.shape[1]] = current

    expected_logits = policy(current)
    actual_logits, _ = model(history)

    torch.testing.assert_close(actual_logits, expected_logits)


def test_policy_value_save_and_load_preserves_configuration(tmp_path: Path) -> None:
    path = tmp_path / "policy_value.pt"
    model = PolicyValueNetwork(
        input_channels=54,
        hidden_size=8,
        dropout=0.2,
        residual_blocks=1,
    )

    save_policy_value(model, path)
    loaded = load_policy_value(path)

    assert loaded.input_channels == 54
    assert loaded.hidden_size == 8
    assert loaded.dropout_rate == 0.2
    assert loaded.residual_blocks == 1


def test_initialize_policy_value_converts_cnn_checkpoint(tmp_path: Path) -> None:
    policy_path = tmp_path / "tactical.pt"
    output_path = tmp_path / "policy_value.pt"
    save_policy(
        ConvolutionalPolicy(hidden_size=8, dropout=0.0, residual_blocks=1),
        policy_path,
    )

    initialize_policy_value(
        policy_path=policy_path,
        output_path=output_path,
        history_length=3,
    )
    loaded = load_policy_value(output_path)

    assert loaded.input_channels == history_observation_shape(3)[0]
