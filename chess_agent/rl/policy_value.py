from pathlib import Path

import torch
from torch import nn

from chess_agent.rl.actions import ACTION_SIZE
from chess_agent.rl.policy import ConvolutionalPolicy, ResidualBlock


class PolicyValueNetwork(nn.Module):
    """Shared chess CNN with policy logits and a scalar state-value output."""

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.residual_blocks = residual_blocks
        self.input_block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            *(ResidualBlock(hidden_size, dropout) for _ in range(residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * 8 * 8, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_size, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        board_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(self.input_block(board_observation.float()))
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


def policy_value_config(model: PolicyValueNetwork) -> dict[str, int | float]:
    return {
        "input_channels": model.input_channels,
        "hidden_size": model.hidden_size,
        "dropout": model.dropout_rate,
        "residual_blocks": model.residual_blocks,
    }


def create_policy_value_from_policy(
    policy: ConvolutionalPolicy,
    *,
    input_channels: int,
) -> PolicyValueNetwork:
    """Create a policy-value model and transfer a tactical CNN policy."""
    model = PolicyValueNetwork(
        input_channels=input_channels,
        hidden_size=policy.hidden_size,
        dropout=policy.dropout_rate,
        residual_blocks=policy.residual_blocks,
    )
    transfer_policy_weights(policy=policy, model=model)
    return model


def transfer_policy_weights(
    *,
    policy: ConvolutionalPolicy,
    model: PolicyValueNetwork,
) -> None:
    """Copy the shared trunk and policy head, padding new input channels with zero."""
    if policy.hidden_size != model.hidden_size:
        raise ValueError("policy and policy-value hidden sizes do not match")
    if policy.residual_blocks != model.residual_blocks:
        raise ValueError("policy and policy-value residual block counts do not match")

    source_state = policy.state_dict()
    target_state = model.state_dict()
    input_weight_key = "input_block.0.weight"
    source_input_weight = source_state[input_weight_key]
    target_input_weight = target_state[input_weight_key]
    if target_input_weight.shape[1] < source_input_weight.shape[1]:
        raise ValueError("policy-value model has fewer input channels than the policy")

    target_input_weight.zero_()
    target_input_weight[:, : source_input_weight.shape[1]].copy_(source_input_weight)

    for key, source_value in source_state.items():
        if key == input_weight_key:
            continue
        target_value = target_state.get(key)
        if target_value is None or target_value.shape != source_value.shape:
            raise ValueError(f"incompatible policy parameter: {key}")
        target_value.copy_(source_value)

    model.load_state_dict(target_state)


def save_policy_value(model: PolicyValueNetwork, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "policy_value",
            "model_config": policy_value_config(model),
            "state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path


def load_policy_value(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PolicyValueNetwork:
    checkpoint = torch.load(Path(path), map_location=device)
    if checkpoint.get("kind") != "policy_value":
        raise ValueError("not a policy-value checkpoint")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("policy-value checkpoint has no model_config")
    model = PolicyValueNetwork(
        input_channels=int(config["input_channels"]),
        hidden_size=int(config.get("hidden_size", 64)),
        dropout=float(config.get("dropout", 0.1)),
        residual_blocks=int(config.get("residual_blocks", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
