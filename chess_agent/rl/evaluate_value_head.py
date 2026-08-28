import argparse
from pathlib import Path

import torch

from chess_agent.rl.policy_value import load_policy_value
from chess_agent.rl.pretrain_value_head import ValueMetrics, evaluate_value_head
from chess_agent.rl.value_dataset import (
    ValueDatasetSummary,
    load_value_dataset,
    summarize_value_dataset,
)


def evaluate_value_checkpoint(
    *,
    model_path: str | Path,
    data_path: str | Path,
    batch_size: int,
    device: str,
) -> tuple[ValueMetrics, ValueDatasetSummary]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_device = resolve_device(device)
    dataset = load_value_dataset(data_path)
    model = load_policy_value(model_path, device=resolved_device)
    if dataset.observation_shape != (model.input_channels, 8, 8):
        raise ValueError("value dataset observation shape does not match the model")
    metrics = evaluate_value_head(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=resolved_device,
    )
    return metrics, summarize_value_dataset(dataset)


def build_value_evaluation_report(
    *,
    model_path: str | Path,
    data_path: str | Path,
    metrics: ValueMetrics,
    dataset: ValueDatasetSummary,
) -> str:
    metadata = dataset.metadata
    lines = [
        "Value-head dataset evaluation",
        "=============================",
        f"Model:          {model_path}",
        f"Dataset:        {data_path}",
        f"Opponent:       {metadata.get('opponent', 'unknown')}",
        f"Max plies:      {metadata.get('max_plies', 'unknown')}",
        f"Gamma:          {metadata.get('gamma', 'unknown')}",
        f"Positions:      {dataset.positions}",
        f"Games:          {dataset.games}",
        f"W/D/L:          {dataset.wins}/{dataset.draws}/{dataset.losses}",
        f"Target mean:    {dataset.target_mean:.5f}",
        f"Target std:     {dataset.target_std:.5f}",
        "",
        "Metrics",
        "-------",
        f"Huber loss:     {metrics.loss:.5f}",
        f"MAE:            {metrics.mae:.5f}",
        f"RMSE:           {metrics.rmse:.5f}",
        f"Explained var:  {metrics.explained_variance:.5f}",
        f"Prediction std: {metrics.prediction_std:.5f}",
        "",
    ]
    return "\n".join(lines)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    metrics, dataset = evaluate_value_checkpoint(
        model_path=args.model_path,
        data_path=args.data,
        batch_size=args.batch_size,
        device=args.device,
    )
    report = build_value_evaluation_report(
        model_path=args.model_path,
        data_path=args.data,
        metrics=metrics,
        dataset=dataset,
    )
    print(report, end="")
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(report, encoding="utf-8")
        print(f"Saved report:   {args.output_path}")


if __name__ == "__main__":
    main()
