import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from chess_agent.rl.random_baseline import random_action_from_mask
from chess_agent.rl.tactical_puzzle_env import TacticalPuzzleEnv


@dataclass(frozen=True)
class TacticalEvaluationResult:
    episodes: int
    successes: int
    illegal_actions: int
    correct_moves: int
    expected_moves: int
    total_reward: float

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes

    @property
    def move_accuracy(self) -> float:
        if self.expected_moves == 0:
            return 0.0
        return self.correct_moves / self.expected_moves

    @property
    def average_reward(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.total_reward / self.episodes


@torch.no_grad()
def evaluate_tactical_policy(
    *,
    policy,
    env: TacticalPuzzleEnv | None = None,
    episodes: int = 100,
    device: str | torch.device | None = None,
) -> TacticalEvaluationResult:
    from chess_agent.rl.train_mate_in_one import greedy_action

    if episodes < 0:
        raise ValueError("episodes must be non-negative")

    env = env or TacticalPuzzleEnv()
    if device is None:
        device = next(policy.parameters()).device
    else:
        device = torch.device(device)
        policy = policy.to(device)

    was_training = policy.training
    policy.eval()
    successes = 0
    illegal_actions = 0
    correct_moves = 0
    expected_moves = 0
    total_reward = 0.0

    for episode in range(episodes):
        observation, info = env.reset(options={"puzzle_index": episode % len(env.puzzles)})
        expected_moves += int(info["total_agent_moves"])
        terminated = False
        truncated = False
        last_info = info
        while not terminated and not truncated:
            action = greedy_action(policy, observation, device)
            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            correct_moves += int(last_info.get("is_correct", False))

        successes += int(last_info.get("is_success", False))
        illegal_actions += int(last_info.get("illegal_action", False))

    policy.train(was_training)
    return TacticalEvaluationResult(
        episodes=episodes,
        successes=successes,
        illegal_actions=illegal_actions,
        correct_moves=correct_moves,
        expected_moves=expected_moves,
        total_reward=total_reward,
    )


def evaluate_tactical_random_baseline(
    *,
    env: TacticalPuzzleEnv | None = None,
    episodes: int = 100,
    seed: int | None = 0,
) -> TacticalEvaluationResult:
    if episodes < 0:
        raise ValueError("episodes must be non-negative")

    import numpy as np

    env = env or TacticalPuzzleEnv()
    rng = np.random.default_rng(seed)
    successes = 0
    illegal_actions = 0
    correct_moves = 0
    expected_moves = 0
    total_reward = 0.0

    for episode in range(episodes):
        observation, info = env.reset(
            seed=seed if episode == 0 else None,
            options={"puzzle_index": episode % len(env.puzzles)},
        )
        expected_moves += int(info["total_agent_moves"])
        terminated = False
        truncated = False
        last_info = info
        while not terminated and not truncated:
            action = random_action_from_mask(observation["action_mask"], rng)
            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            correct_moves += int(last_info.get("is_correct", False))

        successes += int(last_info.get("is_success", False))
        illegal_actions += int(last_info.get("illegal_action", False))

    return TacticalEvaluationResult(
        episodes=episodes,
        successes=successes,
        illegal_actions=illegal_actions,
        correct_moves=correct_moves,
        expected_moves=expected_moves,
        total_reward=total_reward,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "policy"], default="random")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--puzzles-file", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = TacticalPuzzleEnv(puzzles_file=args.puzzles_file)
    if args.agent == "random":
        result = evaluate_tactical_random_baseline(
            env=env,
            episodes=args.episodes,
            seed=args.seed,
        )
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required for --agent policy")
        result = evaluate_saved_tactical_policy(
            env=env,
            model_path=args.model_path,
            episodes=args.episodes,
            device=args.device,
        )

    print_result(args.agent, result)


def evaluate_saved_tactical_policy(
    *,
    env: TacticalPuzzleEnv,
    model_path: Path,
    episodes: int,
    device: str,
) -> TacticalEvaluationResult:
    from chess_agent.rl.train_mate_in_one import load_policy

    policy = load_policy(model_path, device=device)
    return evaluate_tactical_policy(
        policy=policy,
        env=env,
        episodes=episodes,
        device=device,
    )


def print_result(agent_name: str, result: TacticalEvaluationResult) -> None:
    print("Tactical puzzle evaluation")
    print(f"Agent:          {agent_name}")
    print(f"Episodes:       {result.episodes}")
    print(f"Successes:      {result.successes}")
    print(f"Success rate:   {result.success_rate:.1%}")
    print(f"Move accuracy:  {result.move_accuracy:.1%}")
    print(f"Illegal moves:  {result.illegal_actions}")
    print(f"Average reward: {result.average_reward:.3f}")


if __name__ == "__main__":
    main()
