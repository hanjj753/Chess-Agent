import numpy as np

from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv
from chess_agent.rl.random_baseline import evaluate_random_baseline, random_action_from_mask


def test_random_action_from_mask_selects_enabled_action() -> None:
    mask = np.array([0, 1, 0, 1], dtype=np.int8)
    rng = np.random.default_rng(0)

    assert random_action_from_mask(mask, rng) in {1, 3}


def test_evaluate_random_baseline_counts_episodes() -> None:
    result = evaluate_random_baseline(
        env=ChessMateInOneEnv(),
        episodes=8,
        seed=0,
    )

    assert result.episodes == 8
    assert 0 <= result.successes <= 8
    assert result.illegal_actions == 0
    assert -1.0 <= result.average_reward <= 1.0
