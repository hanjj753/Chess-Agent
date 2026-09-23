import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
import re

from chess_agent.rl.compare_full_chess_evaluations import (
    compare_evaluation_csvs,
    paired_mean_confidence_interval,
    read_evaluation_games,
)
from chess_agent.rl.evaluate_full_chess_ppo import (
    format_full_chess_report,
    infer_history_length,
    load_game_results_csv,
    save_game_results_csv,
    save_report,
)
from chess_agent.rl.train_full_chess_ppo import (
    PPO_OPPONENTS,
    FullChessEvaluationResult,
    TrackedMaskablePPO,
    evaluate_full_chess_ppo,
)


@dataclass(frozen=True)
class CheckpointSpec:
    step: int
    path: Path


@dataclass(frozen=True)
class CheckpointSeriesRow:
    step: int
    model_path: str
    result: FullChessEvaluationResult
    score_delta: float
    ci_low: float
    ci_high: float


def checkpoint_step(path: str | Path) -> int:
    match = re.search(r"_(\d+)\.zip$", Path(path).name)
    if match is None:
        raise ValueError(
            "checkpoint filename must end with _<timesteps>.zip: "
            f"{Path(path).name}"
        )
    return int(match.group(1))


def discover_checkpoints(
    checkpoint_dir: str | Path,
    *,
    pattern: str = "full_chess_ppo_*.zip",
) -> tuple[CheckpointSpec, ...]:
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {directory}")

    checkpoints = tuple(
        sorted(
            (
                CheckpointSpec(step=checkpoint_step(path), path=path)
                for path in directory.glob(pattern)
                if path.is_file()
            ),
            key=lambda checkpoint: checkpoint.step,
        )
    )
    if not checkpoints:
        raise ValueError(
            f"no checkpoints matched {pattern!r} in {directory}"
        )
    steps = [checkpoint.step for checkpoint in checkpoints]
    if len(steps) != len(set(steps)):
        raise ValueError(f"duplicate checkpoint steps found in {directory}")
    return checkpoints


def evaluate_model(
    model_path: str | Path,
    *,
    games: int,
    opponent: str,
    alpha_move_probability: float,
    opponent_depth: int,
    opponent_time_limit: float | None,
    history_length: int | None,
    max_plies: int,
    deterministic: bool,
    seed: int,
    device: str,
) -> FullChessEvaluationResult:
    model = TrackedMaskablePPO.load(model_path, device=device)
    resolved_history_length = (
        history_length
        if history_length is not None
        else infer_history_length(model)
    )
    return evaluate_full_chess_ppo(
        model=model,
        episodes=games,
        history_length=resolved_history_length,
        max_plies=max_plies,
        opponent=opponent,
        alpha_move_probability=alpha_move_probability,
        opponent_depth=opponent_depth,
        opponent_time_limit=opponent_time_limit,
        deterministic=deterministic,
        seed=seed,
    )


def paired_delta(
    baseline_csv: str | Path,
    candidate_csv: str | Path,
) -> tuple[float, float, float]:
    baseline = read_evaluation_games(baseline_csv)
    candidate = read_evaluation_games(candidate_csv)
    if baseline.keys() != candidate.keys():
        raise ValueError(
            "baseline and checkpoint evaluations must use identical "
            "seed/color pairs"
        )
    deltas = [
        candidate[key].score - baseline[key].score
        for key in sorted(baseline)
    ]
    delta = sum(deltas) / len(deltas)
    ci_low, ci_high = paired_mean_confidence_interval(deltas)
    return delta, ci_low, ci_high


def format_series_report(
    *,
    baseline_path: str | Path,
    baseline_result: FullChessEvaluationResult,
    rows: tuple[CheckpointSeriesRow, ...],
    opponent: str,
    games: int,
    seed: int,
    max_plies: int,
) -> str:
    baseline_terminations = Counter(
        game.termination for game in baseline_result.games
    )
    lines = [
        "Full-chess PPO checkpoint learning curve",
        f"Baseline:       {baseline_path}",
        f"Opponent:       {opponent}",
        f"Games/model:    {games}",
        f"Base seed:      {seed}",
        f"Max plies:      {max_plies}",
        "",
        "Step       W/D/L             Score    Delta (95% CI)          Avg plies  Max-ply",
        (
            f"baseline   {baseline_result.wins:4d}/"
            f"{baseline_result.draws:4d}/{baseline_result.losses:4d}  "
            f"{baseline_result.score_rate:7.2%}  "
            f"{'-':24s}  {baseline_result.average_plies:9.1f}  "
            f"{baseline_terminations.get('max_plies', 0):7d}"
        ),
    ]
    for row in rows:
        terminations = Counter(game.termination for game in row.result.games)
        interval = (
            f"{row.score_delta:+.2%} "
            f"[{row.ci_low:+.2%}, {row.ci_high:+.2%}]"
        )
        lines.append(
            f"{row.step:8d}   {row.result.wins:4d}/"
            f"{row.result.draws:4d}/{row.result.losses:4d}  "
            f"{row.result.score_rate:7.2%}  {interval:24s}  "
            f"{row.result.average_plies:9.1f}  "
            f"{terminations.get('max_plies', 0):7d}"
        )
    lines.extend(
        [
            "",
            "Delta and confidence intervals are paired against the baseline",
            "using identical game seeds and agent colors.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_series_csv(
    path: str | Path,
    *,
    baseline_path: str | Path,
    baseline_result: FullChessEvaluationResult,
    rows: tuple[CheckpointSeriesRow, ...],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "step",
        "model_path",
        "wins",
        "draws",
        "losses",
        "score_rate",
        "score_delta",
        "ci_low",
        "ci_high",
        "average_reward",
        "average_plies",
        "max_plies_terminations",
        "checkmates",
        "illegal_actions",
    )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        write_series_row(
            writer,
            step=0,
            model_path=str(baseline_path),
            result=baseline_result,
            score_delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
        )
        for row in rows:
            write_series_row(
                writer,
                step=row.step,
                model_path=row.model_path,
                result=row.result,
                score_delta=row.score_delta,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
            )
    return output_path


def write_series_row(
    writer: csv.DictWriter,
    *,
    step: int,
    model_path: str,
    result: FullChessEvaluationResult,
    score_delta: float,
    ci_low: float,
    ci_high: float,
) -> None:
    terminations = Counter(game.termination for game in result.games)
    writer.writerow(
        {
            "step": step,
            "model_path": model_path,
            "wins": result.wins,
            "draws": result.draws,
            "losses": result.losses,
            "score_rate": result.score_rate,
            "score_delta": score_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "average_reward": result.average_reward,
            "average_plies": result.average_plies,
            "max_plies_terminations": terminations.get("max_plies", 0),
            "checkmates": terminations.get("checkmate", 0),
            "illegal_actions": result.illegal_actions,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-model-path", type=Path)
    baseline.add_argument("--baseline-games-csv", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-pattern",
        default="full_chess_ppo_*.zip",
    )
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--opponent", choices=PPO_OPPONENTS, default="random")
    parser.add_argument("--alpha-move-probability", type=float, default=0.1)
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--opponent-time-limit", type=float)
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun evaluations even when per-checkpoint CSV files exist",
    )
    args = parser.parse_args()

    if args.games < 1:
        parser.error("--games must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(
        args.checkpoint_dir,
        pattern=args.checkpoint_pattern,
    )

    if args.baseline_games_csv is not None:
        baseline_csv = args.baseline_games_csv
        baseline_result = load_game_results_csv(baseline_csv)
        baseline_path: Path = baseline_csv
    else:
        assert args.baseline_model_path is not None
        baseline_path = args.baseline_model_path
        baseline_report_path = args.output_dir / "baseline.txt"
        baseline_csv = args.output_dir / "baseline_games.csv"
        if baseline_csv.exists() and not args.force:
            print(f"Reusing baseline: {baseline_csv}", flush=True)
            baseline_result = load_game_results_csv(baseline_csv)
        else:
            print(f"Evaluating baseline: {baseline_path}", flush=True)
            baseline_result = evaluate_model(
                baseline_path,
                games=args.games,
                opponent=args.opponent,
                alpha_move_probability=args.alpha_move_probability,
                opponent_depth=args.opponent_depth,
                opponent_time_limit=args.opponent_time_limit,
                history_length=args.history_length,
                max_plies=args.max_plies,
                deterministic=not args.stochastic,
                seed=args.seed,
                device=args.device,
            )
            save_game_results_csv(
                baseline_csv,
                result=baseline_result,
                base_seed=args.seed,
            )
        save_report(
            baseline_report_path,
            format_full_chess_report(
                model_path=baseline_path,
                opponent=args.opponent,
                result=baseline_result,
                seed=args.seed,
                deterministic=not args.stochastic,
                max_plies=args.max_plies,
                alpha_move_probability=args.alpha_move_probability,
            ),
        )
    if baseline_result.episodes != args.games:
        parser.error(
            "cached baseline evaluation row count does not match --games"
        )

    rows: list[CheckpointSeriesRow] = []
    for checkpoint in checkpoints:
        stem = f"checkpoint_{checkpoint.step:08d}"
        report_path = args.output_dir / f"{stem}.txt"
        games_path = args.output_dir / f"{stem}_games.csv"
        comparison_path = args.output_dir / f"{stem}_vs_baseline.txt"
        if games_path.exists() and not args.force:
            print(f"Reusing step {checkpoint.step}: {games_path}", flush=True)
            result = load_game_results_csv(games_path)
        else:
            print(
                f"Evaluating step {checkpoint.step}: {checkpoint.path}",
                flush=True,
            )
            result = evaluate_model(
                checkpoint.path,
                games=args.games,
                opponent=args.opponent,
                alpha_move_probability=args.alpha_move_probability,
                opponent_depth=args.opponent_depth,
                opponent_time_limit=args.opponent_time_limit,
                history_length=args.history_length,
                max_plies=args.max_plies,
                deterministic=not args.stochastic,
                seed=args.seed,
                device=args.device,
            )
            save_game_results_csv(
                games_path,
                result=result,
                base_seed=args.seed,
            )
        if result.episodes != args.games:
            raise ValueError(
                f"cached evaluation has {result.episodes} games, "
                f"expected {args.games}: {games_path}"
            )
        save_report(
            report_path,
            format_full_chess_report(
                model_path=checkpoint.path,
                opponent=args.opponent,
                result=result,
                seed=args.seed,
                deterministic=not args.stochastic,
                max_plies=args.max_plies,
                alpha_move_probability=args.alpha_move_probability,
            ),
        )
        score_delta, ci_low, ci_high = paired_delta(
            baseline_csv,
            games_path,
        )
        save_report(
            comparison_path,
            compare_evaluation_csvs(baseline_csv, games_path),
        )
        rows.append(
            CheckpointSeriesRow(
                step=checkpoint.step,
                model_path=str(checkpoint.path),
                result=result,
                score_delta=score_delta,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )

    series_rows = tuple(rows)
    report = format_series_report(
        baseline_path=baseline_path,
        baseline_result=baseline_result,
        rows=series_rows,
        opponent=args.opponent,
        games=args.games,
        seed=args.seed,
        max_plies=args.max_plies,
    )
    report_path = save_report(args.output_dir / "checkpoint_curve.txt", report)
    csv_path = save_series_csv(
        args.output_dir / "checkpoint_curve.csv",
        baseline_path=baseline_path,
        baseline_result=baseline_result,
        rows=series_rows,
    )
    print()
    print(report, end="")
    print(f"Saved summary:  {report_path}")
    print(f"Saved CSV:      {csv_path}")


if __name__ == "__main__":
    main()
