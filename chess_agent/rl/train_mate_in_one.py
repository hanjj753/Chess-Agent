import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributions import Categorical

from chess_agent.rl.checkpoints import (
    load_training_checkpoint,
    move_optimizer_state_to_device,
    policy_from_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv
from chess_agent.rl.policy import (
    MateInOnePolicy,
    PolicyNetwork,
    apply_action_mask,
    policy_config,
    policy_from_config,
)
from chess_agent.rl.random_baseline import EvaluationResult


POLICY_GRADIENT_CHECKPOINT_KIND = "mate_in_one_policy_gradient"


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 2_000
    learning_rate: float = 1e-3
    hidden_size: int = 256
    seed: int = 0
    log_every: int = 100
    device: str = "cpu"
    puzzles_file: Path | None = None
    evaluation_puzzles_file: Path | None = None
    evaluation_episodes: int | None = None
    pretrained_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_every: int = 0
    resume_from: Path | None = None


@dataclass(frozen=True)
class TrainingResult:
    episodes: int
    successes: int
    total_reward: float
    final_evaluation: EvaluationResult

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes


def train_policy_gradient(
    *,
    config: TrainingConfig,
    env: ChessMateInOneEnv | None = None,
    policy: MateInOnePolicy | None = None,
) -> tuple[MateInOnePolicy, TrainingResult]:
    if config.episodes < 0:
        raise ValueError("episodes must be non-negative")
    if config.evaluation_episodes is not None and config.evaluation_episodes < 0:
        raise ValueError("evaluation_episodes must be non-negative")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if config.checkpoint_every and config.checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpoint_every is set")
    if config.resume_from is not None and config.pretrained_path is not None:
        raise ValueError("use resume_from or pretrained_path, not both")
    if config.resume_from is not None and policy is not None:
        raise ValueError("use resume_from or policy, not both")

    torch.manual_seed(config.seed)
    env = env or ChessMateInOneEnv(puzzles_file=config.puzzles_file)
    evaluation_env = (
        ChessMateInOneEnv(puzzles_file=config.evaluation_puzzles_file)
        if config.evaluation_puzzles_file is not None
        else env
    )
    evaluation_episodes = (
        config.evaluation_episodes
        if config.evaluation_episodes is not None
        else len(evaluation_env.puzzles)
    )
    device = torch.device(config.device)
    completed_episodes = 0
    successes = 0
    total_reward = 0.0
    if config.resume_from is not None:
        checkpoint = load_training_checkpoint(
            config.resume_from,
            expected_kind=POLICY_GRADIENT_CHECKPOINT_KIND,
        )
        policy = policy_from_checkpoint(checkpoint, device=device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        restore_rng_state(checkpoint, device=device)

        progress = checkpoint.get("progress", {})
        completed_episodes = int(progress.get("completed_episodes", 0))
        successes = int(progress.get("successes", 0))
        total_reward = float(progress.get("total_reward", 0.0))
        if completed_episodes > config.episodes:
            raise ValueError(
                "checkpoint has already completed more episodes than requested"
            )
    else:
        if policy is None and config.pretrained_path is not None:
            policy = load_policy(config.pretrained_path, device=device)
        policy = (policy or MateInOnePolicy(hidden_size=config.hidden_size)).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    for episode in range(completed_episodes + 1, config.episodes + 1):
        observation, _ = env.reset(
            seed=config.seed if episode == 1 else None,
            options={"puzzle_index": (episode - 1) % len(env.puzzles)},
        )
        logits = masked_logits_for_observation(policy, observation, device)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        _, reward, _, _, info = env.step(int(action.item()))

        reward_tensor = torch.tensor(float(reward), device=device)
        loss = -distribution.log_prob(action) * reward_tensor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        successes += int(info.get("is_checkmate", False))
        total_reward += float(reward)

        if config.log_every and episode % config.log_every == 0:
            evaluation = evaluate_policy(
                policy=policy,
                env=evaluation_env,
                episodes=evaluation_episodes,
                device=device,
            )
            print(
                f"episode={episode:5d} "
                f"train_success={successes / episode:.1%} "
                f"eval_success={evaluation.success_rate:.1%} "
                f"avg_reward={total_reward / episode:.3f}",
                flush=True,
            )
        if should_save_checkpoint(
            checkpoint_path=config.checkpoint_path,
            checkpoint_every=config.checkpoint_every,
            step=episode,
        ):
            save_policy_gradient_checkpoint(
                config.checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                completed_episodes=episode,
                successes=successes,
                total_reward=total_reward,
            )

    final_evaluation = evaluate_policy(
        policy=policy,
        env=evaluation_env,
        episodes=evaluation_episodes,
        device=device,
    )
    if config.checkpoint_path is not None:
        save_policy_gradient_checkpoint(
            config.checkpoint_path,
            policy=policy,
            optimizer=optimizer,
            completed_episodes=config.episodes,
            successes=successes,
            total_reward=total_reward,
        )
    return policy, TrainingResult(
        episodes=config.episodes,
        successes=successes,
        total_reward=total_reward,
        final_evaluation=final_evaluation,
    )


def should_save_checkpoint(
    *,
    checkpoint_path: Path | None,
    checkpoint_every: int,
    step: int,
) -> bool:
    return (
        checkpoint_path is not None
        and checkpoint_every > 0
        and step > 0
        and step % checkpoint_every == 0
    )


def save_policy_gradient_checkpoint(
    path: str | Path,
    *,
    policy: MateInOnePolicy,
    optimizer: torch.optim.Optimizer,
    completed_episodes: int,
    successes: int,
    total_reward: float,
) -> Path:
    return save_training_checkpoint(
        path,
        kind=POLICY_GRADIENT_CHECKPOINT_KIND,
        policy=policy,
        optimizer=optimizer,
        progress={
            "completed_episodes": completed_episodes,
            "successes": successes,
            "total_reward": total_reward,
        },
    )


@torch.no_grad()
def evaluate_policy(
    *,
    policy: MateInOnePolicy,
    env: ChessMateInOneEnv | None = None,
    episodes: int = 100,
    device: str | torch.device | None = None,
) -> EvaluationResult:
    if episodes < 0:
        raise ValueError("episodes must be non-negative")

    env = env or ChessMateInOneEnv()
    if device is None:
        device = next(policy.parameters()).device
    else:
        device = torch.device(device)
        policy = policy.to(device)

    was_training = policy.training
    policy.eval()
    successes = 0
    illegal_actions = 0
    total_reward = 0.0

    for episode in range(episodes):
        observation, _ = env.reset(options={"puzzle_index": episode % len(env.puzzles)})
        action = greedy_action(policy, observation, device)
        _, reward, _, _, info = env.step(action)
        successes += int(info.get("is_checkmate", False))
        illegal_actions += int(info.get("illegal_action", False))
        total_reward += float(reward)

    policy.train(was_training)
    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        illegal_actions=illegal_actions,
        total_reward=total_reward,
    )


def greedy_action(
    policy: MateInOnePolicy,
    observation: dict[str, object],
    device: torch.device,
) -> int:
    logits = masked_logits_for_observation(policy, observation, device)
    return int(torch.argmax(logits, dim=-1).item())


def masked_logits_for_observation(
    policy: MateInOnePolicy,
    observation: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    board = torch.as_tensor(
        observation["board"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    action_mask = torch.as_tensor(
        observation["action_mask"],
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    logits = policy(board)
    return apply_action_mask(logits, action_mask)


def save_policy(policy: PolicyNetwork, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_size": policy.hidden_size,
            "policy_config": policy_config(policy),
            "state_dict": policy.state_dict(),
        },
        output_path,
    )
    return output_path


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> PolicyNetwork:
    checkpoint = torch.load(Path(path), map_location=device)
    config = checkpoint.get("policy_config")
    if not isinstance(config, dict):
        config = {"architecture": "mlp", "hidden_size": checkpoint["hidden_size"]}
    policy = policy_from_config(config)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--puzzles-file", type=Path)
    parser.add_argument("--evaluation-puzzles-file", type=Path)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--pretrained-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    policy, result = train_policy_gradient(
        config=TrainingConfig(
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            seed=args.seed,
            log_every=args.log_every,
            device=args.device,
            puzzles_file=args.puzzles_file,
            evaluation_puzzles_file=args.evaluation_puzzles_file,
            evaluation_episodes=args.evaluation_episodes,
            pretrained_path=args.pretrained_path,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume_from,
        )
    )

    print()
    print("Training summary")
    print(f"Episodes:             {result.episodes}")
    print(f"Training success:     {result.success_rate:.1%}")
    print(f"Final eval success:   {result.final_evaluation.success_rate:.1%}")
    print(f"Final average reward: {result.final_evaluation.average_reward:.3f}")

    if args.save_path is not None:
        saved_path = save_policy(policy, args.save_path)
        print(f"Saved policy:         {saved_path}")


if __name__ == "__main__":
    main()
