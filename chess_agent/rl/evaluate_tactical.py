import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from chess_agent.rl.random_baseline import random_action_from_mask
from chess_agent.rl.tactical_puzzle_env import TacticalPuzzle, TacticalPuzzleEnv


RATING_BUCKET_SIZE = 200


@dataclass(frozen=True)
class TacticalBreakdownRow:
    label: str
    episodes: int
    successes: int
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


@dataclass(frozen=True)
class TacticalEvaluationResult:
    episodes: int
    successes: int
    illegal_actions: int
    correct_moves: int
    expected_moves: int
    total_reward: float
    rating_breakdown: tuple[TacticalBreakdownRow, ...] = ()
    theme_breakdown: tuple[TacticalBreakdownRow, ...] = ()
    move_count_breakdown: tuple[TacticalBreakdownRow, ...] = ()

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
    breakdowns = TacticalBreakdownAccumulator()

    for episode in range(episodes):
        puzzle_index = episode % len(env.puzzles)
        puzzle = env.puzzles[puzzle_index]
        observation, info = env.reset(options={"puzzle_index": puzzle_index})
        episode_expected_moves = int(info["total_agent_moves"])
        episode_correct_moves = 0
        episode_reward = 0.0
        expected_moves += episode_expected_moves
        terminated = False
        truncated = False
        last_info = info
        while not terminated and not truncated:
            action = greedy_action(policy, observation, device)
            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            episode_reward += float(reward)
            is_correct = int(last_info.get("is_correct", False))
            correct_moves += is_correct
            episode_correct_moves += is_correct

        is_success = int(last_info.get("is_success", False))
        successes += is_success
        illegal_actions += int(last_info.get("illegal_action", False))
        breakdowns.add(
            puzzle=puzzle,
            success=is_success,
            correct_moves=episode_correct_moves,
            expected_moves=episode_expected_moves,
            total_reward=episode_reward,
        )

    policy.train(was_training)
    rating_breakdown, theme_breakdown, move_count_breakdown = breakdowns.freeze()
    return TacticalEvaluationResult(
        episodes=episodes,
        successes=successes,
        illegal_actions=illegal_actions,
        correct_moves=correct_moves,
        expected_moves=expected_moves,
        total_reward=total_reward,
        rating_breakdown=rating_breakdown,
        theme_breakdown=theme_breakdown,
        move_count_breakdown=move_count_breakdown,
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
    breakdowns = TacticalBreakdownAccumulator()

    for episode in range(episodes):
        puzzle_index = episode % len(env.puzzles)
        puzzle = env.puzzles[puzzle_index]
        observation, info = env.reset(
            seed=seed if episode == 0 else None,
            options={"puzzle_index": puzzle_index},
        )
        episode_expected_moves = int(info["total_agent_moves"])
        episode_correct_moves = 0
        episode_reward = 0.0
        expected_moves += episode_expected_moves
        terminated = False
        truncated = False
        last_info = info
        while not terminated and not truncated:
            action = random_action_from_mask(observation["action_mask"], rng)
            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            episode_reward += float(reward)
            is_correct = int(last_info.get("is_correct", False))
            correct_moves += is_correct
            episode_correct_moves += is_correct

        is_success = int(last_info.get("is_success", False))
        successes += is_success
        illegal_actions += int(last_info.get("illegal_action", False))
        breakdowns.add(
            puzzle=puzzle,
            success=is_success,
            correct_moves=episode_correct_moves,
            expected_moves=episode_expected_moves,
            total_reward=episode_reward,
        )

    rating_breakdown, theme_breakdown, move_count_breakdown = breakdowns.freeze()
    return TacticalEvaluationResult(
        episodes=episodes,
        successes=successes,
        illegal_actions=illegal_actions,
        correct_moves=correct_moves,
        expected_moves=expected_moves,
        total_reward=total_reward,
        rating_breakdown=rating_breakdown,
        theme_breakdown=theme_breakdown,
        move_count_breakdown=move_count_breakdown,
    )


@dataclass
class MutableTacticalStats:
    episodes: int = 0
    successes: int = 0
    correct_moves: int = 0
    expected_moves: int = 0
    total_reward: float = 0.0

    def add(
        self,
        *,
        success: int,
        correct_moves: int,
        expected_moves: int,
        total_reward: float,
    ) -> None:
        self.episodes += 1
        self.successes += success
        self.correct_moves += correct_moves
        self.expected_moves += expected_moves
        self.total_reward += total_reward

    def freeze(self, label: str) -> TacticalBreakdownRow:
        return TacticalBreakdownRow(
            label=label,
            episodes=self.episodes,
            successes=self.successes,
            correct_moves=self.correct_moves,
            expected_moves=self.expected_moves,
            total_reward=self.total_reward,
        )


class TacticalBreakdownAccumulator:
    def __init__(self) -> None:
        self.by_rating: dict[int | None, MutableTacticalStats] = {}
        self.by_theme: dict[str, MutableTacticalStats] = {}
        self.by_move_count: dict[int, MutableTacticalStats] = {}

    def add(
        self,
        *,
        puzzle: TacticalPuzzle,
        success: int,
        correct_moves: int,
        expected_moves: int,
        total_reward: float,
    ) -> None:
        rating_key = rating_bucket_start(puzzle.rating)
        add_group_result(
            self.by_rating,
            rating_key,
            success=success,
            correct_moves=correct_moves,
            expected_moves=expected_moves,
            total_reward=total_reward,
        )
        add_group_result(
            self.by_move_count,
            puzzle.agent_move_count,
            success=success,
            correct_moves=correct_moves,
            expected_moves=expected_moves,
            total_reward=total_reward,
        )
        for theme in set(puzzle.themes) or {"unknown"}:
            add_group_result(
                self.by_theme,
                theme,
                success=success,
                correct_moves=correct_moves,
                expected_moves=expected_moves,
                total_reward=total_reward,
            )

    def freeze(
        self,
    ) -> tuple[
        tuple[TacticalBreakdownRow, ...],
        tuple[TacticalBreakdownRow, ...],
        tuple[TacticalBreakdownRow, ...],
    ]:
        rating_rows = tuple(
            self.by_rating[key].freeze(rating_bucket_label(key))
            for key in sorted(
                self.by_rating,
                key=lambda value: (value is None, value if value is not None else 0),
            )
        )
        theme_rows = tuple(
            self.by_theme[theme].freeze(theme)
            for theme in sorted(self.by_theme)
        )
        move_count_rows = tuple(
            self.by_move_count[count].freeze(str(count))
            for count in sorted(self.by_move_count)
        )
        return rating_rows, theme_rows, move_count_rows


def add_group_result(
    groups: dict,
    key,
    *,
    success: int,
    correct_moves: int,
    expected_moves: int,
    total_reward: float,
) -> None:
    groups.setdefault(key, MutableTacticalStats()).add(
        success=success,
        correct_moves=correct_moves,
        expected_moves=expected_moves,
        total_reward=total_reward,
    )


def rating_bucket_start(rating: int | None) -> int | None:
    if rating is None:
        return None
    return (rating // RATING_BUCKET_SIZE) * RATING_BUCKET_SIZE


def rating_bucket_label(bucket_start: int | None) -> str:
    if bucket_start is None:
        return "unknown"
    return f"{bucket_start}-{bucket_start + RATING_BUCKET_SIZE - 1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "policy"], default="random")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--puzzles-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--min-theme-episodes",
        type=int,
        default=20,
        help="hide theme rows with fewer episodes; use 0 to show all",
    )
    args = parser.parse_args()
    if args.min_theme_episodes < 0:
        raise ValueError("--min-theme-episodes must be non-negative")

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

    print_result(
        args.agent,
        result,
        min_theme_episodes=args.min_theme_episodes,
    )


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


def print_result(
    agent_name: str,
    result: TacticalEvaluationResult,
    *,
    min_theme_episodes: int = 20,
) -> None:
    print("Tactical puzzle evaluation")
    print(f"Agent:          {agent_name}")
    print(f"Episodes:       {result.episodes}")
    print(f"Successes:      {result.successes}")
    print(f"Success rate:   {result.success_rate:.1%}")
    print(f"Move accuracy:  {result.move_accuracy:.1%}")
    print(f"Illegal moves:  {result.illegal_actions}")
    print(f"Average reward: {result.average_reward:.3f}")
    print_breakdown("Rating breakdown", result.rating_breakdown)
    print_breakdown("Agent move-count breakdown", result.move_count_breakdown)

    visible_themes = tuple(
        row for row in result.theme_breakdown
        if row.episodes >= min_theme_episodes
    )
    print_breakdown(
        f"Theme breakdown (multi-label, min episodes={min_theme_episodes})",
        tuple(
            sorted(
                visible_themes,
                key=lambda row: (row.success_rate, -row.episodes, row.label),
            )
        ),
    )


def print_breakdown(
    title: str,
    rows: tuple[TacticalBreakdownRow, ...],
) -> None:
    print()
    print(title)
    if not rows:
        print("No groups to display.")
        return

    label_width = max(len("Group"), *(len(row.label) for row in rows))
    print(
        f"{'Group':<{label_width}}  "
        f"{'Episodes':>8}  "
        f"{'Success':>8}  "
        f"{'Move acc':>8}  "
        f"{'Avg reward':>10}"
    )
    for row in rows:
        print(
            f"{row.label:<{label_width}}  "
            f"{row.episodes:8d}  "
            f"{row.success_rate:8.1%}  "
            f"{row.move_accuracy:8.1%}  "
            f"{row.average_reward:10.3f}"
        )


if __name__ == "__main__":
    main()
