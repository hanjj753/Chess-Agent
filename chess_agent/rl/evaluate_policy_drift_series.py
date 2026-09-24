import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from chess_agent.rl.evaluate_checkpoint_series import discover_checkpoints
from chess_agent.rl.evaluate_full_chess_ppo import save_report
from chess_agent.rl.evaluate_policy_drift import (
    PolicyDriftResult,
    evaluate_policy_drift,
    format_policy_drift_report,
    save_positions_csv,
)
from chess_agent.rl.evaluate_tactical import parse_episode_count
from chess_agent.rl.tactical_puzzle_env import load_tactical_puzzles
from chess_agent.rl.train_full_chess_ppo import TrackedMaskablePPO


@dataclass(frozen=True)
class PolicyDriftSeriesRow:
    absolute_step: int
    candidate_model_path: str
    result: PolicyDriftResult

    @property
    def accuracy_delta(self) -> float:
        return self.result.candidate_accuracy - self.result.reference_accuracy

    @property
    def correct_probability_delta(self) -> float:
        return (
            self.result.mean_candidate_correct_probability
            - self.result.mean_reference_correct_probability
        )

    @property
    def entropy_delta(self) -> float:
        return (
            self.result.mean_candidate_entropy
            - self.result.mean_reference_entropy
        )


def format_policy_drift_series_report(
    *,
    reference_model_path: str | Path,
    puzzles_file: str | Path,
    baseline_step: int,
    rows: tuple[PolicyDriftSeriesRow, ...],
) -> str:
    if not rows:
        raise ValueError("at least one policy drift row is required")
    reference = rows[0].result
    lines = [
        "PPO checkpoint policy drift curve",
        f"Reference:       {reference_model_path}",
        f"Puzzles file:    {puzzles_file}",
        f"Puzzles:         {reference.puzzles}",
        f"Agent positions: {len(reference.positions)}",
        f"Baseline step:   {baseline_step}",
        "",
        "Absolute  Added       KL       JS     Agree   Accuracy  Acc delta  Correct P  P delta  Ent delta",
        (
            f"{baseline_step:8d} {0:6d}  {0.0:8.6f} {0.0:8.6f} "
            f"{1.0:8.1%} {reference.reference_accuracy:9.1%} "
            f"{0.0:+10.1%} "
            f"{reference.mean_reference_correct_probability:10.1%} "
            f"{0.0:+8.1%} {0.0:+10.4f}"
        ),
    ]
    for row in rows:
        result = row.result
        lines.append(
            f"{row.absolute_step:8d} "
            f"{row.absolute_step - baseline_step:6d}  "
            f"{result.mean_kl_reference_to_candidate:8.6f} "
            f"{result.mean_js_divergence:8.6f} "
            f"{result.top1_agreement_rate:8.1%} "
            f"{result.candidate_accuracy:9.1%} "
            f"{row.accuracy_delta:+10.1%} "
            f"{result.mean_candidate_correct_probability:10.1%} "
            f"{row.correct_probability_delta:+8.1%} "
            f"{row.entropy_delta:+10.4f}"
        )
    lines.extend(
        [
            "",
            "KL/JS and agreement compare each checkpoint with the fixed reference.",
            "Negative accuracy or correct-probability deltas indicate tactical retention loss.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_policy_drift_series_csv(
    path: str | Path,
    *,
    reference_model_path: str | Path,
    baseline_step: int,
    rows: tuple[PolicyDriftSeriesRow, ...],
) -> Path:
    if not rows:
        raise ValueError("at least one policy drift row is required")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "absolute_step",
        "added_timesteps",
        "model_path",
        "mean_kl_reference_to_candidate",
        "mean_js_divergence",
        "top1_agreement_rate",
        "entropy",
        "entropy_delta",
        "accuracy",
        "accuracy_delta",
        "correct_probability",
        "correct_probability_delta",
        "candidate_only_correct",
        "reference_only_correct",
        "puzzles",
        "agent_positions",
    )
    reference = rows[0].result
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "absolute_step": baseline_step,
                "added_timesteps": 0,
                "model_path": str(reference_model_path),
                "mean_kl_reference_to_candidate": 0.0,
                "mean_js_divergence": 0.0,
                "top1_agreement_rate": 1.0,
                "entropy": reference.mean_reference_entropy,
                "entropy_delta": 0.0,
                "accuracy": reference.reference_accuracy,
                "accuracy_delta": 0.0,
                "correct_probability": (
                    reference.mean_reference_correct_probability
                ),
                "correct_probability_delta": 0.0,
                "candidate_only_correct": 0,
                "reference_only_correct": 0,
                "puzzles": reference.puzzles,
                "agent_positions": len(reference.positions),
            }
        )
        for row in rows:
            result = row.result
            writer.writerow(
                {
                    "absolute_step": row.absolute_step,
                    "added_timesteps": row.absolute_step - baseline_step,
                    "model_path": row.candidate_model_path,
                    "mean_kl_reference_to_candidate": (
                        result.mean_kl_reference_to_candidate
                    ),
                    "mean_js_divergence": result.mean_js_divergence,
                    "top1_agreement_rate": result.top1_agreement_rate,
                    "entropy": result.mean_candidate_entropy,
                    "entropy_delta": row.entropy_delta,
                    "accuracy": result.candidate_accuracy,
                    "accuracy_delta": row.accuracy_delta,
                    "correct_probability": (
                        result.mean_candidate_correct_probability
                    ),
                    "correct_probability_delta": (
                        row.correct_probability_delta
                    ),
                    "candidate_only_correct": result.candidate_only_correct,
                    "reference_only_correct": result.reference_only_correct,
                    "puzzles": result.puzzles,
                    "agent_positions": len(result.positions),
                }
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-pattern",
        default="full_chess_ppo_*.zip",
    )
    parser.add_argument("--baseline-step", type=int, required=True)
    parser.add_argument("--puzzles-file", type=Path, required=True)
    parser.add_argument(
        "--puzzles",
        type=parse_episode_count,
        default=1_000,
        help="number of puzzles to probe, or 'all'",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-positions",
        action="store_true",
        help="also save one detailed position CSV per checkpoint",
    )
    args = parser.parse_args()

    if args.baseline_step < 0:
        parser.error("--baseline-step must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    checkpoints = discover_checkpoints(
        args.checkpoint_dir,
        pattern=args.checkpoint_pattern,
    )
    if any(checkpoint.step <= args.baseline_step for checkpoint in checkpoints):
        parser.error("checkpoint steps must be greater than --baseline-step")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    puzzles = load_tactical_puzzles(args.puzzles_file)
    reference_model = TrackedMaskablePPO.load(
        args.reference_model_path,
        device=args.device,
    )
    rows: list[PolicyDriftSeriesRow] = []
    for checkpoint in checkpoints:
        print(
            f"Evaluating policy drift at step {checkpoint.step}: "
            f"{checkpoint.path}",
            flush=True,
        )
        candidate_model = TrackedMaskablePPO.load(
            checkpoint.path,
            device=args.device,
        )
        result = evaluate_policy_drift(
            reference_model=reference_model,
            candidate_model=candidate_model,
            puzzles=puzzles,
            max_puzzles=args.puzzles,
            batch_size=args.batch_size,
        )
        stem = f"policy_drift_{checkpoint.step:08d}"
        save_report(
            args.output_dir / f"{stem}.txt",
            format_policy_drift_report(
                reference_model_path=args.reference_model_path,
                candidate_model_path=checkpoint.path,
                puzzles_file=args.puzzles_file,
                result=result,
            ),
        )
        if args.save_positions:
            save_positions_csv(
                args.output_dir / f"{stem}_positions.csv",
                result=result,
            )
        rows.append(
            PolicyDriftSeriesRow(
                absolute_step=checkpoint.step,
                candidate_model_path=str(checkpoint.path),
                result=result,
            )
        )

    series_rows = tuple(rows)
    report = format_policy_drift_series_report(
        reference_model_path=args.reference_model_path,
        puzzles_file=args.puzzles_file,
        baseline_step=args.baseline_step,
        rows=series_rows,
    )
    report_path = save_report(args.output_dir / "policy_drift_curve.txt", report)
    csv_path = save_policy_drift_series_csv(
        args.output_dir / "policy_drift_curve.csv",
        reference_model_path=args.reference_model_path,
        baseline_step=args.baseline_step,
        rows=series_rows,
    )
    print()
    print(report, end="")
    print(f"Saved summary:  {report_path}")
    print(f"Saved CSV:      {csv_path}")


if __name__ == "__main__":
    main()
