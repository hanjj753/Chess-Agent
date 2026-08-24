from pathlib import Path
from typing import Any

import torch

from chess_agent.rl.policy import PolicyNetwork, policy_config, policy_from_config


def save_training_checkpoint(
    path: str | Path,
    *,
    kind: str,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    progress: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": kind,
            "hidden_size": policy.hidden_size,
            "policy_config": policy_config(policy),
            "state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "progress": progress,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_training_checkpoint(path: str | Path, *, expected_kind: str) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    kind = checkpoint.get("kind")
    if kind != expected_kind:
        raise ValueError(f"expected {expected_kind} checkpoint, got: {kind}")
    return checkpoint


def policy_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> PolicyNetwork:
    config = checkpoint.get("policy_config")
    if not isinstance(config, dict):
        config = {"architecture": "mlp", "hidden_size": checkpoint["hidden_size"]}
    policy = policy_from_config(config)
    policy.load_state_dict(checkpoint["state_dict"])
    return policy.to(device)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_rng_state(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> None:
    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())

    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
