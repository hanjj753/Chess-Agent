import csv
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_agent.rl.actions import ACTION_SIZE, move_to_action
from chess_agent.rl.evaluate_policy_drift import (
    compare_position_probabilities,
    evaluate_policy_drift,
    format_policy_drift_report,
    iter_tactical_policy_probes,
    save_positions_csv,
)
from chess_agent.rl.observations import history_observation_shape
from chess_agent.rl.ppo_policy import ChessMaskableActorCriticPolicy
from chess_agent.rl.tactical_puzzle_env import TacticalPuzzle
from chess_agent.rl.train_full_chess_ppo import (
    FullChessPPOConfig,
    TrackedMaskablePPO,
    make_vector_env,
)


PUZZLE = TacticalPuzzle(
    initial_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    line_uci=("e7e5", "g1f3", "b8c6"),
    rating=900,
    themes=("opening", "short"),
)


def test_tactical_probes_include_each_agent_position_and_history() -> None:
    probes = tuple(
        iter_tactical_policy_probes((PUZZLE,), history_length=1)
    )

    assert len(probes) == 2
    assert [probe.expected_move_uci for probe in probes] == ["e7e5", "b8c6"]
    assert [probe.agent_move_index for probe in probes] == [0, 1]
    assert all(
        probe.observation.shape == history_observation_shape(1)
        for probe in probes
    )
    assert all(
        probe.action_mask[
            move_to_action(chess.Move.from_uci(probe.expected_move_uci))
        ]
        for probe in probes
    )
    assert np.count_nonzero(probes[0].observation[18:]) == 0
    assert np.count_nonzero(probes[1].observation[18:]) > 0


def test_probability_comparison_measures_diffusion_and_retention() -> None:
    probe = next(iter_tactical_policy_probes((PUZZLE,), history_length=0))
    expected_action = move_to_action(chess.Move.from_uci("e7e5"))
    alternative_action = move_to_action(chess.Move.from_uci("d7d5"))
    reference = np.zeros(ACTION_SIZE, dtype=np.float64)
    candidate = np.zeros(ACTION_SIZE, dtype=np.float64)
    reference[expected_action] = 0.9
    reference[alternative_action] = 0.1
    candidate[expected_action] = 0.6
    candidate[alternative_action] = 0.4

    position = compare_position_probabilities(
        probe,
        reference_probs=reference,
        candidate_probs=candidate,
    )

    assert position.reference_correct
    assert position.candidate_correct
    assert position.top1_agreement
    assert position.reference_correct_probability == pytest.approx(0.9)
    assert position.candidate_correct_probability == pytest.approx(0.6)
    assert position.candidate_entropy > position.reference_entropy
    assert position.kl_reference_to_candidate > 0
    assert position.js_divergence > 0


def test_identical_ppo_models_have_zero_policy_drift(tmp_path: Path) -> None:
    config = FullChessPPOConfig(
        total_timesteps=0,
        n_envs=1,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        history_length=1,
        max_plies=2,
        hidden_size=8,
        residual_blocks=1,
        evaluation_every=0,
        checkpoint_every=0,
        device="cpu",
        experiment_dir=None,
    )
    env = make_vector_env(config)
    try:
        model = TrackedMaskablePPO(
            ChessMaskableActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={
                "hidden_size": 8,
                "dropout": 0.0,
                "residual_blocks": 1,
            },
            device="cpu",
        )
        result = evaluate_policy_drift(
            reference_model=model,
            candidate_model=model,
            puzzles=(PUZZLE,),
            batch_size=1,
        )
    finally:
        env.close()

    assert result.puzzles == 1
    assert len(result.positions) == 2
    assert result.history_length == 1
    assert result.mean_kl_reference_to_candidate == pytest.approx(0.0)
    assert result.mean_js_divergence == pytest.approx(0.0)
    assert result.top1_agreement_rate == pytest.approx(1.0)
    assert result.mean_candidate_entropy == pytest.approx(
        result.mean_reference_entropy
    )
    assert result.mean_candidate_correct_probability == pytest.approx(
        result.mean_reference_correct_probability
    )

    report = format_policy_drift_report(
        reference_model_path="tmp/reference.zip",
        candidate_model_path="tmp/candidate.zip",
        puzzles_file="data/valid.txt",
        result=result,
    )
    assert "KL(reference || candidate): 0.000000" in report
    assert "Top-1 agreement:           100.0%" in report

    csv_path = save_positions_csv(tmp_path / "positions.csv", result=result)
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert all(row["top1_agreement"] == "1" for row in rows)
