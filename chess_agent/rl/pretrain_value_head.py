import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chess_agent.rl.experiment_tracking import ExperimentLogger
from chess_agent.rl.policy_value import (
    PolicyValueNetwork,
    load_policy_value,
    save_policy_value,
)
from chess_agent.rl.value_dataset import (
    PackedValueDataset,
    ValueDatasetSummary,
    load_value_dataset,
    summarize_value_dataset,
)


@dataclass(frozen=True)
class ValuePretrainingConfig:
    model_path: Path
    train_data_path: Path
    validation_data_path: Path
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-4
    balance_games: bool = True
    balance_outcomes: bool = True
    seed: int = 0
    device: str = "auto"
    save_path: Path = Path("tmp/full_chess_policy_value_pretrained.pt")
    best_model_path: Path = Path("tmp/full_chess_policy_value_pretrained_best.pt")
    experiment_dir: Path | None = Path("analysis/experiments")
    experiment_name: str = "value_head_pretrain"


@dataclass(frozen=True)
class ValueMetrics:
    loss: float
    mae: float
    rmse: float
    explained_variance: float
    target_std: float
    prediction_std: float


@dataclass(frozen=True)
class ValuePretrainingResult:
    completed_epochs: int
    stopped_early: bool
    best_epoch: int
    initial_validation: ValueMetrics
    best_validation: ValueMetrics
    final_validation: ValueMetrics
    final_model_path: Path
    best_model_path: Path
    experiment_run_dir: Path | None
    train_dataset: ValueDatasetSummary
    validation_dataset: ValueDatasetSummary


def pretrain_value_head(
    config: ValuePretrainingConfig,
) -> tuple[PolicyValueNetwork, ValuePretrainingResult]:
    validate_config(config)
    resolved_device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    train_data = load_value_dataset(config.train_data_path)
    validation_data = load_value_dataset(config.validation_data_path)
    validate_dataset_pair(train_data, validation_data)
    train_dataset_summary = summarize_value_dataset(train_data)
    validation_dataset_summary = summarize_value_dataset(validation_data)
    print_dataset_summary("Train", train_dataset_summary)
    print_dataset_summary("Validation", validation_dataset_summary)
    warn_about_outcome_balancing(
        train_dataset_summary,
        enabled=config.balance_outcomes,
    )
    model = load_policy_value(config.model_path, device=resolved_device)
    if tuple(train_data.observation_shape) != (model.input_channels, 8, 8):
        raise ValueError("value dataset observation shape does not match the model")
    freeze_policy_modules(model)
    optimizer = torch.optim.AdamW(
        model.value_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sample_weights = make_sample_weights(
        train_data,
        balance_games=config.balance_games,
        balance_outcomes=config.balance_outcomes,
    )
    experiment_logger = (
        ExperimentLogger.create(
            config.experiment_dir,
            experiment_name=config.experiment_name,
            config=config,
        )
        if config.experiment_dir is not None
        else None
    )
    if experiment_logger is not None:
        print(f"Experiment log: {experiment_logger.run_dir}", flush=True)

    initial_validation = evaluate_value_head(
        model=model,
        dataset=validation_data,
        batch_size=config.batch_size,
        device=resolved_device,
    )
    best_validation = initial_validation
    best_epoch = 0
    best_model_path = save_policy_value(model, config.best_model_path)
    if experiment_logger is not None:
        log_epoch_metrics(
            experiment_logger,
            epoch=0,
            train_loss=None,
            validation=initial_validation,
        )
        experiment_logger.log_checkpoint(
            step=0,
            path=best_model_path,
            is_best=True,
            metrics={"validation_loss": initial_validation.loss},
        )
    print(
        "epoch=  0 "
        f"val_loss={initial_validation.loss:.5f} "
        f"val_mae={initial_validation.mae:.4f} "
        f"val_ev={initial_validation.explained_variance:.4f} "
        f"pred_std={initial_validation.prediction_std:.4f}",
        flush=True,
    )

    completed_epochs = 0
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataset=train_data,
            sample_weights=sample_weights,
            optimizer=optimizer,
            batch_size=config.batch_size,
            device=resolved_device,
            rng=rng,
        )
        validation = evaluate_value_head(
            model=model,
            dataset=validation_data,
            batch_size=config.batch_size,
            device=resolved_device,
        )
        completed_epochs = epoch
        improved = validation.loss < best_validation.loss - config.min_delta
        if improved:
            best_validation = validation
            best_epoch = epoch
            best_model_path = save_policy_value(model, config.best_model_path)
            if experiment_logger is not None:
                experiment_logger.log_checkpoint(
                    step=epoch,
                    path=best_model_path,
                    is_best=True,
                    metrics={"validation_loss": validation.loss},
                )
        if experiment_logger is not None:
            log_epoch_metrics(
                experiment_logger,
                epoch=epoch,
                train_loss=train_loss,
                validation=validation,
            )
        print(
            f"epoch={epoch:3d} train_loss={train_loss:.5f} "
            f"val_loss={validation.loss:.5f} "
            f"val_mae={validation.mae:.4f} "
            f"val_ev={validation.explained_variance:.4f} "
            f"pred_std={validation.prediction_std:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if config.patience > 0 and epoch - best_epoch >= config.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: no validation loss improvement "
                f"for {config.patience} epochs (best epoch: {best_epoch}).",
                flush=True,
            )
            break

    model.eval()
    final_model_path = save_policy_value(model, config.save_path)
    final_validation = evaluate_value_head(
        model=model,
        dataset=validation_data,
        batch_size=config.batch_size,
        device=resolved_device,
    )
    result = ValuePretrainingResult(
        completed_epochs=completed_epochs,
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        initial_validation=initial_validation,
        best_validation=best_validation,
        final_validation=final_validation,
        final_model_path=final_model_path,
        best_model_path=best_model_path,
        experiment_run_dir=(
            experiment_logger.run_dir if experiment_logger is not None else None
        ),
        train_dataset=train_dataset_summary,
        validation_dataset=validation_dataset_summary,
    )
    if experiment_logger is not None:
        experiment_logger.log_checkpoint(
            step=completed_epochs,
            path=final_model_path,
        )
        experiment_logger.save_summary(result)
    return model, result


def freeze_policy_modules(model: PolicyValueNetwork) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.value_head.parameters():
        parameter.requires_grad = True
    model.eval()


def train_one_epoch(
    *,
    model: PolicyValueNetwork,
    dataset: PackedValueDataset,
    sample_weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.eval()
    model.value_head.train()
    indices = rng.permutation(len(dataset))
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        observations = torch.from_numpy(dataset.unpack(batch_indices)).to(device)
        targets = torch.from_numpy(dataset.targets[batch_indices]).to(device)
        weights = torch.from_numpy(sample_weights[batch_indices]).to(device)

        with torch.no_grad():
            features = model.backbone(model.input_block(observations))
        predictions = model.value_head(features).squeeze(-1)
        losses = F.smooth_l1_loss(predictions, targets, reduction="none")
        loss = torch.sum(losses * weights) / torch.sum(weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_weight = float(torch.sum(weights).item())
        weighted_loss_sum += float(torch.sum(losses * weights).item())
        weight_sum += batch_weight
    model.eval()
    return weighted_loss_sum / weight_sum if weight_sum else 0.0


@torch.no_grad()
def evaluate_value_head(
    *,
    model: PolicyValueNetwork,
    dataset: PackedValueDataset,
    batch_size: int,
    device: torch.device,
) -> ValueMetrics:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        observations = torch.from_numpy(dataset.unpack(indices)).to(device)
        features = model.backbone(model.input_block(observations))
        values = model.value_head(features).squeeze(-1)
        predictions.append(values.cpu().numpy())
    predicted = np.concatenate(predictions).astype(np.float64, copy=False)
    targets = dataset.targets.astype(np.float64, copy=False)
    errors = predicted - targets
    mse = float(np.mean(np.square(errors)))
    target_variance = float(np.var(targets))
    explained_variance = (
        1.0 - float(np.var(errors)) / target_variance
        if target_variance > 1e-12
        else 0.0
    )
    return ValueMetrics(
        loss=float(np.mean(F.smooth_l1_loss(
            torch.from_numpy(predicted),
            torch.from_numpy(targets),
            reduction="none",
        ).numpy())),
        mae=float(np.mean(np.abs(errors))),
        rmse=math.sqrt(mse),
        explained_variance=explained_variance,
        target_std=math.sqrt(target_variance),
        prediction_std=float(np.std(predicted)),
    )


def make_sample_weights(
    dataset: PackedValueDataset,
    *,
    balance_games: bool,
    balance_outcomes: bool,
) -> np.ndarray:
    weights = np.ones(len(dataset), dtype=np.float32)
    _, first_indices, game_inverse, game_counts = np.unique(
        dataset.game_ids,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    game_outcomes = dataset.outcomes[first_indices]
    outcome_game_counts = np.bincount(game_outcomes + 1, minlength=3)
    if balance_games:
        weights /= game_counts[game_inverse]
    if balance_outcomes:
        weights /= outcome_game_counts[dataset.outcomes + 1]
    mean_weight = float(np.mean(weights))
    if mean_weight > 0:
        weights /= mean_weight
    return weights


def print_dataset_summary(label: str, summary: ValueDatasetSummary) -> None:
    print(
        f"{label} dataset: positions={summary.positions} games={summary.games} "
        f"W/D/L={summary.wins}/{summary.draws}/{summary.losses} "
        f"target_mean={summary.target_mean:.4f} "
        f"target_std={summary.target_std:.4f}",
        flush=True,
    )


def warn_about_outcome_balancing(
    summary: ValueDatasetSummary,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    outcome_counts = [
        count
        for count in (summary.wins, summary.draws, summary.losses)
        if count > 0
    ]
    if len(outcome_counts) < 2:
        print(
            "Warning: outcome balancing is enabled but the train dataset has "
            "fewer than two observed outcome classes.",
            flush=True,
        )
        return
    if min(outcome_counts) < 0.05 * max(outcome_counts):
        print(
            "Warning: outcome balancing will strongly upweight a rare result "
            "class; consider --no-balance-outcomes for on-policy value data.",
            flush=True,
        )


def log_epoch_metrics(
    logger: ExperimentLogger,
    *,
    epoch: int,
    train_loss: float | None,
    validation: ValueMetrics,
) -> None:
    metrics = {
        "validation_loss": validation.loss,
        "validation_mae": validation.mae,
        "validation_rmse": validation.rmse,
        "validation_explained_variance": validation.explained_variance,
        "validation_target_std": validation.target_std,
        "validation_prediction_std": validation.prediction_std,
    }
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    logger.log_metrics(step=epoch, phase="value_pretrain", metrics=metrics)


def validate_dataset_pair(
    train_data: PackedValueDataset,
    validation_data: PackedValueDataset,
) -> None:
    if not len(train_data) or not len(validation_data):
        raise ValueError("train and validation value datasets must be non-empty")
    if train_data.observation_shape != validation_data.observation_shape:
        raise ValueError("train and validation observation shapes do not match")
    if set(np.unique(train_data.game_ids)) & set(np.unique(validation_data.game_ids)):
        raise ValueError("train and validation datasets contain overlapping game IDs")


def validate_config(config: ValuePretrainingConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.patience < 0:
        raise ValueError("patience must be non-negative")
    if config.min_delta < 0:
        raise ValueError("min_delta must be non-negative")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-balance-games", action="store_true")
    parser.add_argument("--no-balance-outcomes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("tmp/full_chess_policy_value_pretrained.pt"),
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=Path("tmp/full_chess_policy_value_pretrained_best.pt"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("analysis/experiments"),
    )
    parser.add_argument("--experiment-name", default="value_head_pretrain")
    args = parser.parse_args()

    _, result = pretrain_value_head(
        ValuePretrainingConfig(
            model_path=args.model_path,
            train_data_path=args.train_data,
            validation_data_path=args.validation_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            balance_games=not args.no_balance_games,
            balance_outcomes=not args.no_balance_outcomes,
            seed=args.seed,
            device=args.device,
            save_path=args.save_path,
            best_model_path=args.best_model_path,
            experiment_dir=args.experiment_dir,
            experiment_name=args.experiment_name,
        )
    )
    print()
    print("Value-head supervised pretraining summary")
    print(f"Completed epochs:      {result.completed_epochs}")
    print(f"Stopped early:         {'yes' if result.stopped_early else 'no'}")
    print(f"Best epoch:            {result.best_epoch}")
    print(f"Initial val loss:      {result.initial_validation.loss:.5f}")
    print(f"Best val loss:         {result.best_validation.loss:.5f}")
    print(f"Best val EV:           {result.best_validation.explained_variance:.4f}")
    print(f"Best prediction std:   {result.best_validation.prediction_std:.4f}")
    print(f"Final model:           {result.final_model_path}")
    print(f"Best model:            {result.best_model_path}")
    if result.experiment_run_dir is not None:
        print(f"Experiment log:        {result.experiment_run_dir}")


if __name__ == "__main__":
    main()
