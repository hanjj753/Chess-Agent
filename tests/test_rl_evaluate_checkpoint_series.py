import csv
from pathlib import Path

import pytest

from chess_agent.rl.evaluate_checkpoint_series import (
    CheckpointSeriesRow,
    checkpoint_step,
    discover_checkpoints,
    format_series_report,
    paired_delta,
    save_series_csv,
)
from chess_agent.rl.evaluate_full_chess_ppo import save_game_results_csv
from chess_agent.rl.train_full_chess_ppo import (
    FullChessEvaluationResult,
    FullChessGameEvaluation,
)


def make_result(rewards: tuple[float, ...]) -> FullChessEvaluationResult:
    games = []
    for episode, reward in enumerate(rewards, start=1):
        games.append(
            FullChessGameEvaluation(
                episode=episode,
                result=(
                    "1-0"
                    if reward > 0
                    else "0-1" if reward < 0 else "1/2-1/2"
                ),
                reward=reward,
                plies=40 + episode,
                agent_color="white" if episode % 2 else "black",
                termination="checkmate" if reward else "max_plies",
                illegal_action=False,
            )
        )
    return FullChessEvaluationResult(games=tuple(games))


def test_discover_checkpoints_sorts_numeric_steps(tmp_path: Path) -> None:
    for name in (
        "full_chess_ppo_12288.zip",
        "full_chess_ppo_4096.zip",
        "full_chess_ppo_8192.zip",
        "unrelated.zip",
    ):
        (tmp_path / name).touch()

    checkpoints = discover_checkpoints(tmp_path)

    assert [checkpoint.step for checkpoint in checkpoints] == [4096, 8192, 12288]
    assert checkpoint_step(checkpoints[-1].path) == 12288


def test_checkpoint_step_rejects_filename_without_step() -> None:
    with pytest.raises(ValueError, match="timesteps"):
        checkpoint_step("model.zip")


def test_series_report_and_csv_include_paired_delta(tmp_path: Path) -> None:
    baseline = make_result((0.0, -1.0, 1.0, 0.0))
    candidate = make_result((1.0, -1.0, 1.0, 0.0))
    baseline_csv = tmp_path / "baseline_games.csv"
    candidate_csv = tmp_path / "candidate_games.csv"
    save_game_results_csv(baseline_csv, result=baseline, base_seed=100)
    save_game_results_csv(candidate_csv, result=candidate, base_seed=100)
    delta, ci_low, ci_high = paired_delta(baseline_csv, candidate_csv)
    row = CheckpointSeriesRow(
        step=28_672,
        model_path="checkpoint_4096.zip",
        result=candidate,
        score_delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
    )

    report = format_series_report(
        baseline_path="baseline.zip",
        baseline_result=baseline,
        rows=(row,),
        opponent="alpha-random",
        games=4,
        seed=100,
        max_plies=200,
        baseline_step=24_576,
    )
    csv_path = save_series_csv(
        tmp_path / "curve.csv",
        baseline_path="baseline.zip",
        baseline_result=baseline,
        rows=(row,),
        baseline_step=24_576,
    )

    assert delta == pytest.approx(0.125)
    assert "  4096" in report
    assert "+12.50%" in report
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["absolute_step"] for row in rows] == ["24576", "28672"]
    assert [row["added_timesteps"] for row in rows] == ["0", "4096"]
    assert float(rows[1]["score_delta"]) == pytest.approx(0.125)
