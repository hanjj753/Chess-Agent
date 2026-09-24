import csv
from pathlib import Path

import pytest

from chess_agent.rl.evaluate_policy_drift import (
    PolicyDriftPosition,
    PolicyDriftResult,
)
from chess_agent.rl.evaluate_policy_drift_series import (
    PolicyDriftSeriesRow,
    format_policy_drift_series_report,
    save_policy_drift_series_csv,
)


def make_result() -> PolicyDriftResult:
    return PolicyDriftResult(
        puzzles=1,
        history_length=4,
        positions=(
            PolicyDriftPosition(
                puzzle_index=0,
                agent_move_index=0,
                fen="fen-1",
                rating=1200,
                themes=("fork",),
                expected_move_uci="e2e4",
                legal_move_count=20,
                reference_top_move_uci="e2e4",
                candidate_top_move_uci="d2d4",
                reference_entropy=1.0,
                candidate_entropy=1.2,
                reference_correct_probability=0.8,
                candidate_correct_probability=0.4,
                kl_reference_to_candidate=0.1,
                js_divergence=0.02,
            ),
            PolicyDriftPosition(
                puzzle_index=0,
                agent_move_index=1,
                fen="fen-2",
                rating=1200,
                themes=("fork",),
                expected_move_uci="g1f3",
                legal_move_count=18,
                reference_top_move_uci="g1f3",
                candidate_top_move_uci="g1f3",
                reference_entropy=1.1,
                candidate_entropy=1.2,
                reference_correct_probability=0.7,
                candidate_correct_probability=0.7,
                kl_reference_to_candidate=0.04,
                js_divergence=0.01,
            ),
        ),
    )


def test_policy_drift_series_report_and_csv_use_relative_steps(
    tmp_path: Path,
) -> None:
    result = make_result()
    row = PolicyDriftSeriesRow(
        absolute_step=28_672,
        candidate_model_path="checkpoint_28672.zip",
        result=result,
    )

    report = format_policy_drift_series_report(
        reference_model_path="baseline.zip",
        puzzles_file="valid.txt",
        baseline_step=24_576,
        rows=(row,),
    )
    output_path = save_policy_drift_series_csv(
        tmp_path / "policy_drift_curve.csv",
        reference_model_path="baseline.zip",
        baseline_step=24_576,
        rows=(row,),
    )

    assert "  4096" in report
    assert "-50.0%" in report
    with output_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [item["absolute_step"] for item in rows] == ["24576", "28672"]
    assert [item["added_timesteps"] for item in rows] == ["0", "4096"]
    assert float(rows[1]["accuracy_delta"]) == pytest.approx(-0.5)
    assert float(rows[1]["correct_probability_delta"]) == pytest.approx(-0.2)
    assert float(rows[1]["top1_agreement_rate"]) == pytest.approx(0.5)
