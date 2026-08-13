import torch
from torch import nn

from chess_agent.rl.actions import ACTION_SIZE
from chess_agent.rl.observations import OBSERVATION_SHAPE

MASKED_LOGIT = -1_000_000_000.0


class MateInOnePolicy(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        input_size = OBSERVATION_SHAPE[0] * OBSERVATION_SHAPE[1] * OBSERVATION_SHAPE[2]
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_SIZE),
        )

    def forward(self, board_observation: torch.Tensor) -> torch.Tensor:
        return self.network(board_observation.float())


def apply_action_mask(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()
    return logits.masked_fill(~action_mask, MASKED_LOGIT)
