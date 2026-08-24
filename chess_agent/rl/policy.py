import torch
from torch import nn

from chess_agent.rl.actions import ACTION_SIZE
from chess_agent.rl.observations import OBSERVATION_SHAPE

MASKED_LOGIT = -1_000_000_000.0
POLICY_ARCHITECTURES = ("mlp", "cnn")


class MateInOnePolicy(nn.Module):
    architecture = "mlp"

    def __init__(self, hidden_size: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        validate_dropout(dropout)
        input_size = OBSERVATION_SHAPE[0] * OBSERVATION_SHAPE[1] * OBSERVATION_SHAPE[2]
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_SIZE),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, board_observation: torch.Tensor) -> torch.Tensor:
        hidden = self.network[:3](board_observation.float())
        return self.network[3](self.dropout(hidden))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.dropout(outputs)
        outputs = self.norm2(self.conv2(outputs))
        return self.activation(outputs + residual)


class ConvolutionalPolicy(nn.Module):
    architecture = "cnn"

    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.1,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        validate_dropout(dropout)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.residual_blocks = residual_blocks
        self.input_block = nn.Sequential(
            nn.Conv2d(
                OBSERVATION_SHAPE[0],
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

    def forward(self, board_observation: torch.Tensor) -> torch.Tensor:
        features = self.input_block(board_observation.float())
        return self.policy_head(self.backbone(features))


PolicyNetwork = MateInOnePolicy | ConvolutionalPolicy


def create_policy(
    *,
    architecture: str = "mlp",
    hidden_size: int = 256,
    dropout: float = 0.0,
    residual_blocks: int = 3,
) -> PolicyNetwork:
    if architecture == "mlp":
        return MateInOnePolicy(hidden_size=hidden_size, dropout=dropout)
    if architecture == "cnn":
        return ConvolutionalPolicy(
            hidden_size=hidden_size,
            dropout=dropout,
            residual_blocks=residual_blocks,
        )
    raise ValueError(f"unsupported policy architecture: {architecture}")


def policy_config(policy: PolicyNetwork) -> dict[str, int | float | str]:
    config: dict[str, int | float | str] = {
        "architecture": policy.architecture,
        "hidden_size": policy.hidden_size,
        "dropout": policy.dropout_rate,
    }
    if isinstance(policy, ConvolutionalPolicy):
        config["residual_blocks"] = policy.residual_blocks
    return config


def policy_from_config(config: dict[str, object]) -> PolicyNetwork:
    return create_policy(
        architecture=str(config.get("architecture", "mlp")),
        hidden_size=int(config.get("hidden_size", 256)),
        dropout=float(config.get("dropout", 0.0)),
        residual_blocks=int(config.get("residual_blocks", 3)),
    )


def validate_dropout(dropout: float) -> None:
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def apply_action_mask(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()
    return logits.masked_fill(~action_mask, MASKED_LOGIT)
