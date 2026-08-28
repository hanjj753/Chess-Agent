import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean, stdev


@dataclass(frozen=True)
class EvaluationGameRow:
    seed: int
    agent_color: str
    reward: float
    score: float
    plies: int
    termination: str

    @property
    def key(self) -> tuple[int, str]:
        return self.seed, self.agent_color

    @property
    def outcome(self) -> str:
        if self.reward > 0:
            return "win"
        if self.reward < 0:
            return "loss"
        return "draw"


def compare_evaluation_csvs(
    first_path: str | Path,
    second_path: str | Path,
) -> str:
    first_rows = read_evaluation_games(first_path)
    second_rows = read_evaluation_games(second_path)
    if first_rows.keys() != second_rows.keys():
        missing_from_second = sorted(first_rows.keys() - second_rows.keys())
        missing_from_first = sorted(second_rows.keys() - first_rows.keys())
        raise ValueError(
            "evaluation files do not contain the same seed/color pairs: "
            f"missing from second={missing_from_second[:5]}, "
            f"missing from first={missing_from_first[:5]}"
        )

    pairs = [(first_rows[key], second_rows[key]) for key in sorted(first_rows)]
    first_scores = [first.score for first, _ in pairs]
    second_scores = [second.score for _, second in pairs]
    deltas = [second - first for first, second in zip(first_scores, second_scores)]
    improved = sum(delta > 0 for delta in deltas)
    unchanged = sum(delta == 0 for delta in deltas)
    worsened = sum(delta < 0 for delta in deltas)
    delta_mean = mean(deltas) if deltas else 0.0
    ci_low, ci_high = paired_mean_confidence_interval(deltas)
    ply_delta = mean(second.plies - first.plies for first, second in pairs) if pairs else 0.0

    lines = [
        "Full-chess paired evaluation comparison",
        f"A: {first_path}",
        f"B: {second_path}",
        f"Paired games: {len(pairs)}",
        "",
        "Aggregate results",
        "Model        W/D/L       Score   Avg plies",
        format_aggregate_row("A", [first for first, _ in pairs]),
        format_aggregate_row("B", [second for _, second in pairs]),
        "",
        "Paired change (B - A)",
        f"Score delta:       {delta_mean:+.2%}",
        f"95% CI:            [{ci_low:+.2%}, {ci_high:+.2%}]",
        f"Improved/same/worse: {improved}/{unchanged}/{worsened}",
        f"Mean ply delta:    {ply_delta:+.2f}",
        f"Sign-test p-value: {two_sided_sign_test(improved, worsened):.4f}",
        "",
        "Outcome transitions",
        "A -> B             Games",
    ]
    transitions: dict[tuple[str, str], int] = {}
    for first, second in pairs:
        key = first.outcome, second.outcome
        transitions[key] = transitions.get(key, 0) + 1
    for (first_outcome, second_outcome), count in sorted(transitions.items()):
        lines.append(f"{first_outcome:5s} -> {second_outcome:5s} {count:8d}")
    lines.append("")
    return "\n".join(lines)


def read_evaluation_games(path: str | Path) -> dict[tuple[int, str], EvaluationGameRow]:
    rows: dict[tuple[int, str], EvaluationGameRow] = {}
    with Path(path).open(encoding="utf-8", newline="") as source:
        for raw in csv.DictReader(source):
            row = EvaluationGameRow(
                seed=int(raw["seed"]),
                agent_color=raw["agent_color"],
                reward=float(raw["reward"]),
                score=float(raw["score"]),
                plies=int(raw["plies"]),
                termination=raw["termination"],
            )
            if row.key in rows:
                raise ValueError(f"duplicate seed/color pair in {path}: {row.key}")
            rows[row.key] = row
    if not rows:
        raise ValueError(f"evaluation CSV is empty: {path}")
    return rows


def format_aggregate_row(label: str, rows: list[EvaluationGameRow]) -> str:
    wins = sum(row.outcome == "win" for row in rows)
    draws = sum(row.outcome == "draw" for row in rows)
    losses = sum(row.outcome == "loss" for row in rows)
    score_rate = mean(row.score for row in rows) if rows else 0.0
    average_plies = mean(row.plies for row in rows) if rows else 0.0
    return (
        f"{label:5s} {wins:5d}/{draws:3d}/{losses:3d} "
        f"{score_rate:10.1%} {average_plies:11.1f}"
    )


def paired_mean_confidence_interval(
    deltas: list[float],
) -> tuple[float, float]:
    if len(deltas) < 2:
        value = deltas[0] if deltas else 0.0
        return value, value
    center = mean(deltas)
    margin = 1.96 * stdev(deltas) / math.sqrt(len(deltas))
    return center - margin, center + margin


def two_sided_sign_test(improved: int, worsened: int) -> float:
    non_ties = improved + worsened
    if non_ties == 0:
        return 1.0
    tail_end = min(improved, worsened)
    tail_probability = sum(
        math.comb(non_ties, count) for count in range(tail_end + 1)
    ) / (2**non_ties)
    return min(1.0, 2.0 * tail_probability)


def save_comparison(path: str | Path, report: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_csv", type=Path)
    parser.add_argument("second_csv", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    report = compare_evaluation_csvs(args.first_csv, args.second_csv)
    print(report, end="")
    if args.output_path is not None:
        saved_path = save_comparison(args.output_path, report)
        print(f"Saved comparison: {saved_path}")


if __name__ == "__main__":
    main()
