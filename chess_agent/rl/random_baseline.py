from dataclasses import dataclass

import numpy as np

from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    successes: int
    illegal_actions: int
    total_reward: float

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes

    @property
    def average_reward(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.total_reward / self.episodes


def random_action_from_mask(
    action_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    legal_actions = np.flatnonzero(action_mask)
    if len(legal_actions) == 0:
        raise ValueError("action_mask has no legal actions")
    return int(rng.choice(legal_actions))


def evaluate_random_baseline(
    *,
    env: ChessMateInOneEnv | None = None,
    episodes: int = 1_000,
    seed: int | None = 0,
    cycle_puzzles: bool = True,
) -> EvaluationResult:
    if episodes < 0:
        raise ValueError("episodes must be non-negative")

    env = env or ChessMateInOneEnv()
    rng = np.random.default_rng(seed)
    successes = 0
    illegal_actions = 0
    total_reward = 0.0

    for episode in range(episodes):
        options = None
        if cycle_puzzles:
            options = {"puzzle_index": episode % len(env.puzzles)}

        observation, _ = env.reset(
            seed=seed if episode == 0 else None,
            options=options,
        )
        action = random_action_from_mask(observation["action_mask"], rng)
        _, reward, _, _, info = env.step(action)

        successes += int(info.get("is_checkmate", False))
        illegal_actions += int(info.get("illegal_action", False))
        total_reward += float(reward)

    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        illegal_actions=illegal_actions,
        total_reward=total_reward,
    )
