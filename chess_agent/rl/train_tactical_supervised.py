import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chess_agent.rl.actions import legal_action_mask, move_to_action
from chess_agent.rl.checkpoints import (
    load_training_checkpoint,
    move_optimizer_state_to_device,
    policy_from_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from chess_agent.rl.evaluate_tactical import evaluate_tactical_policy
from chess_agent.rl.observations import board_to_observation
from chess_agent.rl.policy import (
    POLICY_ARCHITECTURES,
    PolicyNetwork,
    apply_action_mask,
    create_policy,
)
from chess_agent.rl.tactical_puzzle_env import TacticalPuzzle, TacticalPuzzleEnv, load_tactical_puzzles
from chess_agent.rl.train_mate_in_one import load_policy, save_policy
from chess_agent.rl.train_mate_in_one_supervised import AccuracyResult, evaluate_labeled_accuracy


TACTICAL_SUPERVISED_CHECKPOINT_KIND = "tactical_supervised"


@dataclass(frozen=True)
class TacticalTrainingSample:
    fen: str
    target_uci: str
    target_action: int
    puzzle_index: int
    line_index: int
    rating: int | None = None
    themes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TacticalSupervisedTrainingConfig:
    puzzles_file: Path
    validation_file: Path | None = None
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    architecture: str = "cnn"
    hidden_size: int = 64
    dropout: float = 0.1
    residual_blocks: int = 3
    weight_decay: float = 1e-4
    early_stopping_patience: int | None = 15
    early_stopping_min_delta: float = 0.0
    train_fraction: float = 0.9
    seed: int = 0
    device: str = "cpu"
    max_puzzles: int | None = None
    log_every: int = 1
    evaluation_episodes: int | None = None
    save_path: Path | None = None
    load_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_every: int = 0
    best_checkpoint_path: Path | None = None
    resume_from: Path | None = None


@dataclass(frozen=True)
class TacticalSupervisedTrainingResult:
    train_accuracy: AccuracyResult
    validation_accuracy: AccuracyResult
    validation_puzzle_success_rate: float
    validation_move_accuracy: float
    best_validation_accuracy: float
    best_epoch: int | None
    completed_epochs: int
    stopped_early: bool


class TacticalSampleDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: Sequence[TacticalTrainingSample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        board = chess.Board(sample.fen)
        return (
            torch.as_tensor(board_to_observation(board), dtype=torch.float32),
            torch.as_tensor(legal_action_mask(board), dtype=torch.bool),
            torch.tensor(sample.target_action, dtype=torch.long),
        )


def train_tactical_supervised_policy(
    config: TacticalSupervisedTrainingConfig,
) -> tuple[PolicyNetwork, TacticalSupervisedTrainingResult]:
    if config.epochs < 0:
        raise ValueError("epochs must be non-negative")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < config.train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if config.checkpoint_every and config.checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpoint_every is set")
    if config.resume_from is not None and config.load_path is not None:
        raise ValueError("use resume_from or load_path, not both")
    if config.architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"unsupported policy architecture: {config.architecture}")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    source_puzzles = load_tactical_puzzles(config.puzzles_file)
    if config.max_puzzles is not None:
        if config.max_puzzles < 0:
            raise ValueError("max_puzzles must be non-negative")
        source_puzzles = source_puzzles[: config.max_puzzles]

    train_puzzles, validation_puzzles = make_train_validation_puzzles(
        train_puzzles=source_puzzles,
        validation_file=config.validation_file,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    train_dataset = TacticalSampleDataset(samples_from_puzzles(train_puzzles))
    validation_dataset = TacticalSampleDataset(samples_from_puzzles(validation_puzzles))
    if config.early_stopping_patience is not None and len(validation_dataset) == 0:
        raise ValueError("validation puzzles are required for early stopping")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    completed_epochs = 0
    best_validation_accuracy = -1.0
    best_epoch: int | None = None
    epochs_without_improvement = 0
    if config.resume_from is not None:
        checkpoint = load_training_checkpoint(
            config.resume_from,
            expected_kind=TACTICAL_SUPERVISED_CHECKPOINT_KIND,
        )
        policy = policy_from_checkpoint(checkpoint, device=device)
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        restore_rng_state(checkpoint, device=device)
        progress = checkpoint.get("progress", {})
        completed_epochs = int(progress.get("completed_epochs", 0))
        best_validation_accuracy = float(progress.get("best_validation_accuracy", -1.0))
        raw_best_epoch = progress.get("best_epoch")
        best_epoch = int(raw_best_epoch) if raw_best_epoch is not None else None
        epochs_without_improvement = int(
            progress.get("epochs_without_improvement", 0)
        )
        if completed_epochs > config.epochs:
            raise ValueError(
                "checkpoint has already completed more epochs than requested"
            )
    else:
        policy = (
            load_policy(config.load_path, device=device)
            if config.load_path is not None
            else create_policy(
                architecture=config.architecture,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                residual_blocks=config.residual_blocks,
            ).to(device)
        )
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    stopped_early = False
    last_completed_epoch = completed_epochs
    for epoch in range(completed_epochs + 1, config.epochs + 1):
        policy.train()
        for boards, masks, targets in train_loader:
            boards = boards.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            logits = apply_action_mask(policy(boards), masks)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = None
        validation_accuracy = None
        should_log = bool(config.log_every and epoch % config.log_every == 0)
        should_track_validation = (
            config.best_checkpoint_path is not None
            or config.early_stopping_patience is not None
        )
        if should_log or should_track_validation:
            if should_log:
                train_accuracy = evaluate_labeled_accuracy(
                    policy=policy,
                    dataset=train_dataset,
                    batch_size=config.batch_size,
                    device=device,
                )
            validation_accuracy = evaluate_labeled_accuracy(
                policy=policy,
                dataset=validation_dataset,
                batch_size=config.batch_size,
                device=device,
            )

            improved = (
                validation_accuracy.accuracy
                > best_validation_accuracy + config.early_stopping_min_delta
            )
            if improved:
                best_validation_accuracy = validation_accuracy.accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                if config.best_checkpoint_path is not None:
                    save_tactical_supervised_checkpoint(
                        config.best_checkpoint_path,
                        policy=policy,
                        optimizer=optimizer,
                        completed_epochs=epoch,
                        best_validation_accuracy=best_validation_accuracy,
                        best_epoch=best_epoch,
                        epochs_without_improvement=epochs_without_improvement,
                    )
            else:
                epochs_without_improvement += 1

            if should_log and train_accuracy is not None:
                print(
                    f"epoch={epoch:3d} "
                    f"train_acc={train_accuracy.accuracy:.1%} "
                    f"val_acc={validation_accuracy.accuracy:.1%} "
                    f"train_loss={train_accuracy.average_loss:.4f} "
                    f"val_loss={validation_accuracy.average_loss:.4f} "
                    f"best_val_acc={best_validation_accuracy:.1%} "
                    f"best_epoch={format_best_epoch(best_epoch)}",
                    flush=True,
                )

        last_completed_epoch = epoch
        if should_save_checkpoint(
            checkpoint_path=config.checkpoint_path,
            checkpoint_every=config.checkpoint_every,
            step=epoch,
        ):
            save_tactical_supervised_checkpoint(
                config.checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                completed_epochs=epoch,
                best_validation_accuracy=best_validation_accuracy,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: "
                f"validation accuracy did not improve for "
                f"{config.early_stopping_patience} epochs "
                f"(best epoch: {format_best_epoch(best_epoch)}).",
                flush=True,
            )
            break

    train_accuracy = evaluate_labeled_accuracy(
        policy=policy,
        dataset=train_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    validation_accuracy = evaluate_labeled_accuracy(
        policy=policy,
        dataset=validation_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    if (
        validation_accuracy.accuracy
        > best_validation_accuracy + config.early_stopping_min_delta
    ):
        best_validation_accuracy = validation_accuracy.accuracy
        best_epoch = last_completed_epoch
        if config.best_checkpoint_path is not None:
            save_tactical_supervised_checkpoint(
                config.best_checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                completed_epochs=last_completed_epoch,
                best_validation_accuracy=best_validation_accuracy,
                best_epoch=best_epoch,
                epochs_without_improvement=0,
            )
    validation_success_rate, validation_move_accuracy = evaluate_validation_puzzles(
        policy=policy,
        puzzles=validation_puzzles,
        episodes=config.evaluation_episodes,
        device=device,
    )

    if config.save_path is not None:
        save_policy(policy, config.save_path)
    if config.checkpoint_path is not None:
        save_tactical_supervised_checkpoint(
            config.checkpoint_path,
            policy=policy,
            optimizer=optimizer,
            completed_epochs=last_completed_epoch,
            best_validation_accuracy=best_validation_accuracy,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )

    return policy, TacticalSupervisedTrainingResult(
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        validation_puzzle_success_rate=validation_success_rate,
        validation_move_accuracy=validation_move_accuracy,
        best_validation_accuracy=best_validation_accuracy,
        best_epoch=best_epoch,
        completed_epochs=last_completed_epoch,
        stopped_early=stopped_early,
    )


def make_train_validation_puzzles(
    *,
    train_puzzles: Sequence[TacticalPuzzle],
    validation_file: Path | None,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[TacticalPuzzle, ...], tuple[TacticalPuzzle, ...]]:
    train_puzzles = tuple(train_puzzles)
    if validation_file is not None:
        return train_puzzles, load_tactical_puzzles(validation_file)

    train_size = round(len(train_puzzles) * train_fraction)
    train_size = min(max(train_size, 1), len(train_puzzles))
    indices = torch.randperm(
        len(train_puzzles),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    train_indices = set(indices[:train_size])
    return (
        tuple(puzzle for index, puzzle in enumerate(train_puzzles) if index in train_indices),
        tuple(puzzle for index, puzzle in enumerate(train_puzzles) if index not in train_indices),
    )


def samples_from_puzzles(
    puzzles: Sequence[TacticalPuzzle],
) -> tuple[TacticalTrainingSample, ...]:
    samples: list[TacticalTrainingSample] = []
    for puzzle_index, puzzle in enumerate(puzzles):
        board = chess.Board(puzzle.initial_fen)
        for line_index, move_uci in enumerate(puzzle.line_uci):
            move = chess.Move.from_uci(move_uci)
            if line_index % 2 == 0:
                samples.append(
                    TacticalTrainingSample(
                        fen=board.fen(),
                        target_uci=move.uci(),
                        target_action=move_to_action(move),
                        puzzle_index=puzzle_index,
                        line_index=line_index,
                        rating=puzzle.rating,
                        themes=puzzle.themes,
                    )
                )
            board.push(move)
    return tuple(samples)


@torch.no_grad()
def evaluate_validation_puzzles(
    *,
    policy: PolicyNetwork,
    puzzles: Sequence[TacticalPuzzle],
    episodes: int | None,
    device: torch.device,
) -> tuple[float, float]:
    if not puzzles:
        return 0.0, 0.0

    env = TacticalPuzzleEnv(puzzles=puzzles)
    result = evaluate_tactical_policy(
        policy=policy,
        env=env,
        episodes=episodes if episodes is not None else len(puzzles),
        device=device,
    )
    return result.success_rate, result.move_accuracy


def should_save_checkpoint(
    *,
    checkpoint_path: Path | None,
    checkpoint_every: int,
    step: int,
) -> bool:
    return (
        checkpoint_path is not None
        and checkpoint_every > 0
        and step > 0
        and step % checkpoint_every == 0
    )


def save_tactical_supervised_checkpoint(
    path: str | Path,
    *,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    best_validation_accuracy: float = -1.0,
    best_epoch: int | None = None,
    epochs_without_improvement: int = 0,
) -> Path:
    return save_training_checkpoint(
        path,
        kind=TACTICAL_SUPERVISED_CHECKPOINT_KIND,
        policy=policy,
        optimizer=optimizer,
        progress={
            "completed_epochs": completed_epochs,
            "best_validation_accuracy": best_validation_accuracy,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
    )


def format_best_epoch(best_epoch: int | None) -> str:
    return "-" if best_epoch is None else str(best_epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--architecture", choices=POLICY_ARCHITECTURES, default="cnn")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="early stopping patience; use 0 to disable",
    )
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-puzzles", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--best-checkpoint-path", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    _, result = train_tactical_supervised_policy(
        TacticalSupervisedTrainingConfig(
            puzzles_file=args.puzzles_file,
            validation_file=args.validation_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            architecture=args.architecture,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            residual_blocks=args.residual_blocks,
            weight_decay=args.weight_decay,
            early_stopping_patience=None if args.patience == 0 else args.patience,
            early_stopping_min_delta=args.min_delta,
            train_fraction=args.train_fraction,
            seed=args.seed,
            device=args.device,
            max_puzzles=args.max_puzzles,
            log_every=args.log_every,
            evaluation_episodes=args.evaluation_episodes,
            save_path=args.save_path,
            load_path=args.load_path,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            best_checkpoint_path=args.best_checkpoint_path,
            resume_from=args.resume_from,
        )
    )

    print()
    print("Tactical supervised training summary")
    print(f"Train accuracy:              {result.train_accuracy.accuracy:.1%}")
    print(f"Validation accuracy:         {result.validation_accuracy.accuracy:.1%}")
    print(f"Validation puzzle success:   {result.validation_puzzle_success_rate:.1%}")
    print(f"Validation move accuracy:    {result.validation_move_accuracy:.1%}")
    print(f"Best validation accuracy:    {result.best_validation_accuracy:.1%}")
    print(f"Best epoch:                  {format_best_epoch(result.best_epoch)}")
    print(f"Completed epochs:             {result.completed_epochs}")
    print(f"Stopped early:                {'yes' if result.stopped_early else 'no'}")
    if args.save_path is not None:
        print(f"Saved policy:                {args.save_path}")
    if args.best_checkpoint_path is not None:
        print(f"Best checkpoint:             {args.best_checkpoint_path}")


if __name__ == "__main__":
    main()
