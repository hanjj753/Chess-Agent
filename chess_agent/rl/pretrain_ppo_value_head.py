import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from chess_agent.rl.experiment_tracking import ExperimentLogger
from chess_agent.rl.ppo_policy import (
    ChessMaskableActorCriticPolicy,
    ChessPolicyValueExtractor,
)
from chess_agent.rl.pretrain_value_head import (
    ValueMetrics,
    ValuePretrainingResult,
    log_epoch_metrics,
    make_sample_weights,
    print_dataset_summary,
    validate_dataset_pair,
    warn_about_outcome_balancing,
)
from chess_agent.rl.train_full_chess_ppo import (
    TrackedMaskablePPO,
    save_ppo_model,
)
from chess_agent.rl.value_dataset import (
    PackedValueDataset,
    load_value_dataset,
    summarize_value_dataset,
)


@dataclass(frozen=True)
class PPOValuePretrainingConfig:
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
    balance_outcomes: bool = False
    seed: int = 0
    device: str = "auto"
    save_path: Path = Path("tmp/full_chess_ppo_value_pretrained.zip")
    best_model_path: Path = Path("tmp/full_chess_ppo_value_pretrained_best.zip")
    experiment_dir: Path | None = Path("analysis/experiments")
    experiment_name: str = "ppo_value_head_pretrain"


def pretrain_ppo_value_head(
    config: PPOValuePretrainingConfig,
) -> tuple[TrackedMaskablePPO, ValuePretrainingResult]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    train_data = load_value_dataset(config.train_data_path)
    validation_data = load_value_dataset(config.validation_data_path)
    validate_dataset_pair(train_data, validation_data)
    train_summary = summarize_value_dataset(train_data)
    validation_summary = summarize_value_dataset(validation_data)
    print_dataset_summary("Train", train_summary)
    print_dataset_summary("Validation", validation_summary)
    warn_about_outcome_balancing(
        train_summary,
        enabled=config.balance_outcomes,
    )

    model = TrackedMaskablePPO.load(config.model_path, device=device)
    validate_model_dataset_shape(model, train_data)
    critic_modules = critic_modules_from_model(model)
    critic_parameters = tuple(
        parameter
        for module in critic_modules
        for parameter in module.parameters()
    )
    freeze_except_critic(model, critic_parameters)
    model.policy.optimizer.state.clear()
    optimizer = torch.optim.AdamW(
        critic_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sample_weights = make_sample_weights(
        train_data,
        balance_games=config.balance_games,
        balance_outcomes=config.balance_outcomes,
    )
    logger = (
        ExperimentLogger.create(
            config.experiment_dir,
            experiment_name=config.experiment_name,
            config=config,
        )
        if config.experiment_dir is not None
        else None
    )
    if logger is not None:
        print(f"Experiment log: {logger.run_dir}", flush=True)

    initial_validation = evaluate_ppo_value_head(
        model=model,
        dataset=validation_data,
        batch_size=config.batch_size,
        device=device,
    )
    best_validation = initial_validation
    best_epoch = 0
    best_model_path = save_ppo_model(model, config.best_model_path)
    if logger is not None:
        log_epoch_metrics(
            logger,
            epoch=0,
            train_loss=None,
            validation=initial_validation,
        )
        logger.log_checkpoint(
            step=0,
            path=best_model_path,
            is_best=True,
            metrics={"validation_loss": initial_validation.loss},
        )
    print_validation(epoch=0, train_loss=None, validation=initial_validation, best_epoch=0)

    completed_epochs = 0
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataset=train_data,
            sample_weights=sample_weights,
            optimizer=optimizer,
            batch_size=config.batch_size,
            device=device,
            rng=rng,
        )
        validation = evaluate_ppo_value_head(
            model=model,
            dataset=validation_data,
            batch_size=config.batch_size,
            device=device,
        )
        completed_epochs = epoch
        improved = validation.loss < best_validation.loss - config.min_delta
        if improved:
            best_validation = validation
            best_epoch = epoch
            best_model_path = save_ppo_model(model, config.best_model_path)
            if logger is not None:
                logger.log_checkpoint(
                    step=epoch,
                    path=best_model_path,
                    is_best=True,
                    metrics={"validation_loss": validation.loss},
                )
        if logger is not None:
            log_epoch_metrics(
                logger,
                epoch=epoch,
                train_loss=train_loss,
                validation=validation,
            )
        print_validation(
            epoch=epoch,
            train_loss=train_loss,
            validation=validation,
            best_epoch=best_epoch,
        )
        if config.patience > 0 and epoch - best_epoch >= config.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: no validation loss improvement "
                f"for {config.patience} epochs (best epoch: {best_epoch}).",
                flush=True,
            )
            break

    restore_trainable_parameters(model)
    final_model_path = save_ppo_model(model, config.save_path)
    final_validation = evaluate_ppo_value_head(
        model=model,
        dataset=validation_data,
        batch_size=config.batch_size,
        device=device,
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
        experiment_run_dir=logger.run_dir if logger is not None else None,
        train_dataset=train_summary,
        validation_dataset=validation_summary,
    )
    if logger is not None:
        logger.log_checkpoint(step=completed_epochs, path=final_model_path)
        logger.save_summary(result)
    return model, result


def critic_modules_from_model(
    model: TrackedMaskablePPO,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    policy = model.policy
    if not isinstance(policy, ChessMaskableActorCriticPolicy):
        raise TypeError("PPO model does not use ChessMaskableActorCriticPolicy")
    extractor = policy.mlp_extractor
    if not isinstance(extractor, ChessPolicyValueExtractor):
        raise TypeError("PPO policy does not use ChessPolicyValueExtractor")
    return extractor.value_head, policy.value_net


def freeze_except_critic(
    model: TrackedMaskablePPO,
    critic_parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    for parameter in model.policy.parameters():
        parameter.requires_grad = False
    for parameter in critic_parameters:
        parameter.requires_grad = True
    model.policy.set_training_mode(False)


def restore_trainable_parameters(model: TrackedMaskablePPO) -> None:
    for parameter in model.policy.parameters():
        parameter.requires_grad = True
    model.policy.set_training_mode(False)


def ppo_values(
    model: TrackedMaskablePPO,
    observations: torch.Tensor,
) -> torch.Tensor:
    policy = model.policy
    if not isinstance(policy, ChessMaskableActorCriticPolicy):
        raise TypeError("PPO model does not use ChessMaskableActorCriticPolicy")
    extractor = policy.mlp_extractor
    if not isinstance(extractor, ChessPolicyValueExtractor):
        raise TypeError("PPO policy does not use ChessPolicyValueExtractor")
    features = policy.extract_features(observations)
    critic_latent = extractor.forward_critic(features)
    return policy.value_net(critic_latent).flatten()


def train_one_epoch(
    *,
    model: TrackedMaskablePPO,
    dataset: PackedValueDataset,
    sample_weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    indices = rng.permutation(len(dataset))
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        observations = torch.from_numpy(dataset.unpack(batch_indices)).to(device)
        targets = torch.from_numpy(dataset.targets[batch_indices]).to(device)
        weights = torch.from_numpy(sample_weights[batch_indices]).to(device)

        predictions = ppo_values(model, observations)
        losses = F.smooth_l1_loss(predictions, targets, reduction="none")
        loss = torch.sum(losses * weights) / torch.sum(weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        weighted_loss_sum += float(torch.sum(losses * weights).item())
        weight_sum += float(torch.sum(weights).item())
    return weighted_loss_sum / weight_sum if weight_sum else 0.0


@torch.no_grad()
def evaluate_ppo_value_head(
    *,
    model: TrackedMaskablePPO,
    dataset: PackedValueDataset,
    batch_size: int,
    device: torch.device,
) -> ValueMetrics:
    model.policy.set_training_mode(False)
    predictions: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        observations = torch.from_numpy(dataset.unpack(indices)).to(device)
        predictions.append(ppo_values(model, observations).cpu().numpy())
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
    losses = F.smooth_l1_loss(
        torch.from_numpy(predicted),
        torch.from_numpy(targets),
        reduction="none",
    ).numpy()
    return ValueMetrics(
        loss=float(np.mean(losses)),
        mae=float(np.mean(np.abs(errors))),
        rmse=math.sqrt(mse),
        explained_variance=explained_variance,
        target_std=math.sqrt(target_variance),
        prediction_std=float(np.std(predicted)),
    )


def validate_model_dataset_shape(
    model: TrackedMaskablePPO,
    dataset: PackedValueDataset,
) -> None:
    shape = model.observation_space.shape
    if shape is None or tuple(int(value) for value in shape) != dataset.observation_shape:
        raise ValueError("value dataset observation shape does not match the PPO model")


def validate_config(config: PPOValuePretrainingConfig) -> None:
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


def print_validation(
    *,
    epoch: int,
    train_loss: float | None,
    validation: ValueMetrics,
    best_epoch: int,
) -> None:
    train_text = "" if train_loss is None else f"train_loss={train_loss:.5f} "
    print(
        f"epoch={epoch:3d} {train_text}"
        f"val_loss={validation.loss:.5f} "
        f"val_mae={validation.mae:.4f} "
        f"val_ev={validation.explained_variance:.4f} "
        f"pred_std={validation.prediction_std:.4f} "
        f"best_epoch={best_epoch}",
        flush=True,
    )


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
    parser.add_argument("--balance-outcomes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("tmp/full_chess_ppo_value_pretrained.zip"),
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=Path("tmp/full_chess_ppo_value_pretrained_best.zip"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("analysis/experiments"),
    )
    parser.add_argument("--experiment-name", default="ppo_value_head_pretrain")
    args = parser.parse_args()

    _, result = pretrain_ppo_value_head(
        PPOValuePretrainingConfig(
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
            balance_outcomes=args.balance_outcomes,
            seed=args.seed,
            device=args.device,
            save_path=args.save_path,
            best_model_path=args.best_model_path,
            experiment_dir=args.experiment_dir,
            experiment_name=args.experiment_name,
        )
    )
    print()
    print("PPO value-head supervised pretraining summary")
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
