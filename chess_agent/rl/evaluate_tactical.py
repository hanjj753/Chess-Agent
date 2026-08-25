import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

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
class TacticalAdjustedThemeRow:
    label: str
    episodes: int
    successes: int
    expected_successes: float
    correct_moves: int
    expected_correct_moves: float
    expected_moves: int

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes

    @property
    def expected_success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.expected_successes / self.episodes

    @property
    def success_gap(self) -> float:
        return self.success_rate - self.expected_success_rate

    @property
    def move_accuracy(self) -> float:
        if self.expected_moves == 0:
            return 0.0
        return self.correct_moves / self.expected_moves

    @property
    def expected_move_accuracy(self) -> float:
        if self.expected_moves == 0:
            return 0.0
        return self.expected_correct_moves / self.expected_moves

    @property
    def move_accuracy_gap(self) -> float:
        return self.move_accuracy - self.expected_move_accuracy


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
    adjusted_theme_breakdown: tuple[TacticalAdjustedThemeRow, ...] = ()

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
    (
        rating_breakdown,
        theme_breakdown,
        move_count_breakdown,
        adjusted_theme_breakdown,
    ) = breakdowns.freeze()
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
        adjusted_theme_breakdown=adjusted_theme_breakdown,
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

    (
        rating_breakdown,
        theme_breakdown,
        move_count_breakdown,
        adjusted_theme_breakdown,
    ) = breakdowns.freeze()
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
        adjusted_theme_breakdown=adjusted_theme_breakdown,
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


@dataclass
class MutableAdjustedThemeStats:
    episodes: int = 0
    successes: int = 0
    expected_successes: float = 0.0
    correct_moves: int = 0
    expected_correct_moves: float = 0.0
    expected_moves: int = 0

    def add_stratum(
        self,
        *,
        observed: MutableTacticalStats,
        baseline: MutableTacticalStats,
    ) -> None:
        self.episodes += observed.episodes
        self.successes += observed.successes
        self.expected_successes += observed.episodes * (
            baseline.successes / baseline.episodes
        )
        self.correct_moves += observed.correct_moves
        self.expected_correct_moves += observed.expected_moves * (
            baseline.correct_moves / baseline.expected_moves
        )
        self.expected_moves += observed.expected_moves

    def freeze(self, label: str) -> TacticalAdjustedThemeRow:
        return TacticalAdjustedThemeRow(
            label=label,
            episodes=self.episodes,
            successes=self.successes,
            expected_successes=self.expected_successes,
            correct_moves=self.correct_moves,
            expected_correct_moves=self.expected_correct_moves,
            expected_moves=self.expected_moves,
        )


class TacticalBreakdownAccumulator:
    def __init__(self) -> None:
        self.by_rating: dict[int | None, MutableTacticalStats] = {}
        self.by_theme: dict[str, MutableTacticalStats] = {}
        self.by_move_count: dict[int, MutableTacticalStats] = {}
        self.by_difficulty: dict[
            tuple[int | None, int],
            MutableTacticalStats,
        ] = {}
        self.by_theme_difficulty: dict[
            tuple[str, int | None, int],
            MutableTacticalStats,
        ] = {}

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
        difficulty_key = (rating_key, puzzle.agent_move_count)
        add_group_result(
            self.by_difficulty,
            difficulty_key,
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
            add_group_result(
                self.by_theme_difficulty,
                (theme, *difficulty_key),
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
        tuple[TacticalAdjustedThemeRow, ...],
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
        adjusted_theme_rows = self.freeze_adjusted_themes()
        return rating_rows, theme_rows, move_count_rows, adjusted_theme_rows

    def freeze_adjusted_themes(self) -> tuple[TacticalAdjustedThemeRow, ...]:
        adjusted: dict[str, MutableAdjustedThemeStats] = {}
        for (
            theme,
            rating_key,
            move_count,
        ), observed in self.by_theme_difficulty.items():
            baseline = self.by_difficulty[(rating_key, move_count)]
            adjusted.setdefault(theme, MutableAdjustedThemeStats()).add_stratum(
                observed=observed,
                baseline=baseline,
            )
        return tuple(
            adjusted[theme].freeze(theme)
            for theme in sorted(adjusted)
        )


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
    parser.add_argument(
        "--episodes",
        type=parse_episode_count,
        default=None,
        metavar="N|all",
        help="number of puzzles to evaluate; default: all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--puzzles-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-path", type=Path)
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
    episodes = len(env.puzzles) if args.episodes is None else args.episodes
    if args.agent == "random":
        result = evaluate_tactical_random_baseline(
            env=env,
            episodes=episodes,
            seed=args.seed,
        )
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required for --agent policy")
        result = evaluate_saved_tactical_policy(
            env=env,
            model_path=args.model_path,
            episodes=episodes,
            device=args.device,
        )

    print_result(
        args.agent,
        result,
        min_theme_episodes=args.min_theme_episodes,
    )
    if args.output_path is not None:
        saved_path = save_result_report(
            args.output_path,
            agent_name=args.agent,
            result=result,
            min_theme_episodes=args.min_theme_episodes,
        )
        print()
        print(f"Saved report:   {saved_path}")


def parse_episode_count(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        episodes = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episodes must be a non-negative integer or 'all'") from exc
    if episodes < 0:
        raise argparse.ArgumentTypeError("episodes must be non-negative")
    return episodes


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
    file: TextIO | None = None,
) -> None:
    print("Tactical puzzle evaluation", file=file)
    print(f"Agent:          {agent_name}", file=file)
    print(f"Episodes:       {result.episodes}", file=file)
    print(f"Successes:      {result.successes}", file=file)
    print(f"Success rate:   {result.success_rate:.1%}", file=file)
    print(f"Move accuracy:  {result.move_accuracy:.1%}", file=file)
    print(f"Illegal moves:  {result.illegal_actions}", file=file)
    print(f"Average reward: {result.average_reward:.3f}", file=file)
    print_breakdown("Rating breakdown", result.rating_breakdown, file=file)
    print_breakdown(
        "Agent move-count breakdown",
        result.move_count_breakdown,
        file=file,
    )

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
        file=file,
    )

    visible_adjusted_themes = tuple(
        row for row in result.adjusted_theme_breakdown
        if row.episodes >= min_theme_episodes
    )
    print_adjusted_theme_breakdown(
        (
            "Difficulty-adjusted theme breakdown "
            f"(rating + move count, min episodes={min_theme_episodes})"
        ),
        tuple(
            sorted(
                visible_adjusted_themes,
                key=lambda row: (row.success_gap, -row.episodes, row.label),
            )
        ),
        file=file,
    )


def print_breakdown(
    title: str,
    rows: tuple[TacticalBreakdownRow, ...],
    *,
    file: TextIO | None = None,
) -> None:
    print(file=file)
    print(title, file=file)
    if not rows:
        print("No groups to display.", file=file)
        return

    label_width = max(len("Group"), *(len(row.label) for row in rows))
    print(
        f"{'Group':<{label_width}}  "
        f"{'Episodes':>8}  "
        f"{'Success':>8}  "
        f"{'Move acc':>8}  "
        f"{'Avg reward':>10}",
        file=file,
    )
    for row in rows:
        print(
            f"{row.label:<{label_width}}  "
            f"{row.episodes:8d}  "
            f"{row.success_rate:8.1%}  "
            f"{row.move_accuracy:8.1%}  "
            f"{row.average_reward:10.3f}",
            file=file,
        )


def print_adjusted_theme_breakdown(
    title: str,
    rows: tuple[TacticalAdjustedThemeRow, ...],
    *,
    file: TextIO | None = None,
) -> None:
    print(file=file)
    print(title, file=file)
    if not rows:
        print("No groups to display.", file=file)
        return

    label_width = max(len("Group"), *(len(row.label) for row in rows))
    print(
        f"{'Group':<{label_width}}  "
        f"{'Episodes':>8}  "
        f"{'Observed':>8}  "
        f"{'Expected':>8}  "
        f"{'Success gap':>11}  "
        f"{'Move gap':>8}",
        file=file,
    )
    for row in rows:
        print(
            f"{row.label:<{label_width}}  "
            f"{row.episodes:8d}  "
            f"{row.success_rate:8.1%}  "
            f"{row.expected_success_rate:8.1%}  "
            f"{row.success_gap:+11.1%}  "
            f"{row.move_accuracy_gap:+8.1%}",
            file=file,
        )


def save_result_report(
    path: str | Path,
    *,
    agent_name: str,
    result: TacticalEvaluationResult,
    min_theme_episodes: int = 20,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        print_result(
            agent_name,
            result,
            min_theme_episodes=min_theme_episodes,
            file=handle,
        )
    return output_path


if __name__ == "__main__":
    main()
