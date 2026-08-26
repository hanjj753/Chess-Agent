from typing import Any

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from chess_agent.rl.policy import ResidualBlock
from chess_agent.rl.policy_value import PolicyValueNetwork


class ChessBackboneExtractor(BaseFeaturesExtractor):
    """Spatial feature extractor shared by the PPO actor and critic."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        hidden_size: int,
        dropout: float,
        residual_blocks: int,
    ) -> None:
        if len(observation_space.shape) != 3 or observation_space.shape[1:] != (8, 8):
            raise ValueError("chess observations must have shape (channels, 8, 8)")
        input_channels = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=hidden_size * 8 * 8)
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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.input_block(observations.float()))


class ChessPolicyValueExtractor(nn.Module):
    """Build separate actor and critic latent vectors from shared board features."""

    def __init__(self, *, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.latent_dim_pi = 2 * 8 * 8
        self.latent_dim_vf = hidden_size
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_size, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_head(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features)


class ChessMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Maskable SB3 actor-critic policy using the project's residual chess CNN."""

    def __init__(
        self,
        *args: Any,
        hidden_size: int = 64,
        dropout: float = 0.1,
        residual_blocks: int = 3,
        **kwargs: Any,
    ) -> None:
        self.chess_hidden_size = hidden_size
        self.chess_dropout = dropout
        self.chess_residual_blocks = residual_blocks
        kwargs["features_extractor_class"] = ChessBackboneExtractor
        kwargs["features_extractor_kwargs"] = {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "residual_blocks": residual_blocks,
        }
        kwargs["net_arch"] = []
        kwargs["normalize_images"] = False
        kwargs["ortho_init"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ChessPolicyValueExtractor(
            hidden_size=self.chess_hidden_size,
            dropout=self.chess_dropout,
        )


def transfer_policy_value_to_ppo(
    *,
    source: PolicyValueNetwork,
    target: ChessMaskableActorCriticPolicy,
) -> None:
    """Warm-start an SB3 maskable policy from a project PolicyValueNetwork."""
    observation_shape = target.observation_space.shape
    if observation_shape is None or int(observation_shape[0]) != source.input_channels:
        raise ValueError("source and PPO policy input channel counts do not match")
    if target.chess_hidden_size != source.hidden_size:
        raise ValueError("source and PPO policy hidden sizes do not match")
    if target.chess_residual_blocks != source.residual_blocks:
        raise ValueError("source and PPO residual block counts do not match")

    extractor = target.features_extractor
    if not isinstance(extractor, ChessBackboneExtractor):
        raise TypeError("target policy does not use ChessBackboneExtractor")
    actor_critic = target.mlp_extractor
    if not isinstance(actor_critic, ChessPolicyValueExtractor):
        raise TypeError("target policy does not use ChessPolicyValueExtractor")

    extractor.input_block.load_state_dict(source.input_block.state_dict())
    extractor.backbone.load_state_dict(source.backbone.state_dict())
    actor_critic.policy_head.load_state_dict(source.policy_head[:-1].state_dict())
    actor_critic.value_head.load_state_dict(source.value_head[:-2].state_dict())
    target.action_net.load_state_dict(source.policy_head[-1].state_dict())
    target.value_net.load_state_dict(source.value_head[-2].state_dict())
