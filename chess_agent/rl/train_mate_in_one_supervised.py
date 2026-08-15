import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from chess_agent.rl.checkpoints import (
    load_training_checkpoint,
    move_optimizer_state_to_device,
    policy_from_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from chess_agent.rl.actions import move_to_action, legal_action_mask
from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv, is_mate_after_move
from chess_agent.rl.observations import board_to_observation
from chess_agent.rl.policy import MateInOnePolicy, apply_action_mask
from chess_agent.rl.train_mate_in_one import evaluate_policy, load_policy, save_policy


SUPERVISED_CHECKPOINT_KIND = "mate_in_one_supervised"


@dataclass(frozen=True)
class LabeledPuzzle:
    fen: str
    solution_uci: str
    target_action: int
    rating: int | None = None


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    puzzles_file: Path
    validation_file: Path | None = None
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_size: int = 256
    train_fraction: float = 0.9
    seed: int = 0
    device: str = "cpu"
    max_puzzles: int | None = None
    log_every: int = 1
    save_path: Path | None = None
    load_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_every: int = 0
    resume_from: Path | None = None


@dataclass(frozen=True)
class AccuracyResult:
    correct: int
    total: int
    average_loss: float

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


@dataclass(frozen=True)
class SupervisedTrainingResult:
    train_accuracy: AccuracyResult
    validation_accuracy: AccuracyResult
    validation_mate_success_rate: float


class MateInOneDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, puzzles: Sequence[LabeledPuzzle]) -> None:
        self.puzzles = tuple(puzzles)

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        puzzle = self.puzzles[index]
        board = chess.Board(puzzle.fen)
        return (
            torch.as_tensor(board_to_observation(board), dtype=torch.float32),
            torch.as_tensor(legal_action_mask(board), dtype=torch.bool),
            torch.tensor(puzzle.target_action, dtype=torch.long),
        )


def train_supervised_policy(
    config: SupervisedTrainingConfig,
) -> tuple[MateInOnePolicy, SupervisedTrainingResult]:
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

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    puzzles = load_labeled_puzzles(config.puzzles_file, limit=config.max_puzzles)
    dataset = MateInOneDataset(puzzles)
    train_dataset, validation_dataset = make_train_validation_datasets(
        train_dataset=dataset,
        validation_file=config.validation_file,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    completed_epochs = 0
    if config.resume_from is not None:
        checkpoint = load_training_checkpoint(
            config.resume_from,
            expected_kind=SUPERVISED_CHECKPOINT_KIND,
        )
        policy = policy_from_checkpoint(checkpoint, device=device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        restore_rng_state(checkpoint, device=device)
        progress = checkpoint.get("progress", {})
        completed_epochs = int(progress.get("completed_epochs", 0))
        if completed_epochs > config.epochs:
            raise ValueError(
                "checkpoint has already completed more epochs than requested"
            )
    else:
        policy = (
            load_policy(config.load_path, device=device)
            if config.load_path is not None
            else MateInOnePolicy(hidden_size=config.hidden_size).to(device)
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

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

        if config.log_every and epoch % config.log_every == 0:
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
            print(
                f"epoch={epoch:3d} "
                f"train_acc={train_accuracy.accuracy:.1%} "
                f"val_acc={validation_accuracy.accuracy:.1%} "
                f"train_loss={train_accuracy.average_loss:.4f} "
                f"val_loss={validation_accuracy.average_loss:.4f}",
                flush=True,
            )
        if should_save_checkpoint(
            checkpoint_path=config.checkpoint_path,
            checkpoint_every=config.checkpoint_every,
            step=epoch,
        ):
            save_supervised_checkpoint(
                config.checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                completed_epochs=epoch,
            )

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
    validation_mate_success = evaluate_validation_mate_success(
        policy=policy,
        dataset=validation_dataset,
        device=device,
    )

    if config.save_path is not None:
        save_policy(policy, config.save_path)
    if config.checkpoint_path is not None:
        save_supervised_checkpoint(
            config.checkpoint_path,
            policy=policy,
            optimizer=optimizer,
            completed_epochs=config.epochs,
        )

    return policy, SupervisedTrainingResult(
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        validation_mate_success_rate=validation_mate_success,
    )


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


def save_supervised_checkpoint(
    path: str | Path,
    *,
    policy: MateInOnePolicy,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
) -> Path:
    return save_training_checkpoint(
        path,
        kind=SUPERVISED_CHECKPOINT_KIND,
        policy=policy,
        optimizer=optimizer,
        progress={"completed_epochs": completed_epochs},
    )


def load_labeled_puzzles(path: str | Path, *, limit: int | None = None) -> tuple[LabeledPuzzle, ...]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    puzzles = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        puzzles.append(parse_labeled_puzzle_line(line, line_number=line_number))
        if limit is not None and len(puzzles) >= limit:
            break

    if not puzzles:
        raise ValueError(f"no labeled puzzles found in: {path}")
    return tuple(puzzles)


def parse_labeled_puzzle_line(line: str, *, line_number: int) -> LabeledPuzzle:
    parts = line.split("\t")
    fen = parts[0].strip()
    solution_uci = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None
    rating = parse_optional_int(parts[2].strip()) if len(parts) >= 3 else None

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"invalid FEN at line {line_number}: {fen}") from exc

    if solution_uci is None:
        solution = find_first_mate_in_one_move(board)
    else:
        try:
            solution = chess.Move.from_uci(solution_uci)
        except ValueError as exc:
            raise ValueError(
                f"invalid solution UCI at line {line_number}: {solution_uci}"
            ) from exc
        validate_solution_move(board, solution, line_number=line_number)

    return LabeledPuzzle(
        fen=fen,
        solution_uci=solution.uci(),
        target_action=move_to_action(solution),
        rating=rating,
    )


def find_first_mate_in_one_move(board: chess.Board) -> chess.Move:
    for move in board.legal_moves:
        if is_mate_after_move(board, move):
            return move
    raise ValueError(f"position has no mate-in-one move: {board.fen()}")


def validate_solution_move(
    board: chess.Board,
    solution: chess.Move,
    *,
    line_number: int,
) -> None:
    if solution not in board.legal_moves:
        raise ValueError(f"solution is illegal at line {line_number}: {solution.uci()}")
    if not is_mate_after_move(board, solution):
        raise ValueError(
            f"solution is not checkmate at line {line_number}: {solution.uci()}"
        )


def parse_optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def split_dataset(
    dataset: MateInOneDataset,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    train_size = round(len(dataset) * train_fraction)
    train_size = min(max(train_size, 1), len(dataset))
    validation_size = len(dataset) - train_size
    return random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )


def make_train_validation_datasets(
    *,
    train_dataset: MateInOneDataset,
    validation_file: Path | None,
    train_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if validation_file is None:
        return split_dataset(
            train_dataset,
            train_fraction=train_fraction,
            seed=seed,
        )

    validation_dataset = MateInOneDataset(load_labeled_puzzles(validation_file))
    return train_dataset, validation_dataset


@torch.no_grad()
def evaluate_labeled_accuracy(
    *,
    policy: MateInOnePolicy,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> AccuracyResult:
    if len(dataset) == 0:
        return AccuracyResult(correct=0, total=0, average_loss=0.0)

    was_training = policy.training
    policy.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    total = 0
    correct = 0
    total_loss = 0.0

    for boards, masks, targets in loader:
        boards = boards.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        logits = apply_action_mask(policy(boards), masks)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        predictions = torch.argmax(logits, dim=-1)

        total += int(targets.numel())
        correct += int((predictions == targets).sum().item())
        total_loss += float(loss.item())

    policy.train(was_training)
    return AccuracyResult(
        correct=correct,
        total=total,
        average_loss=total_loss / total,
    )


@torch.no_grad()
def evaluate_validation_mate_success(
    *,
    policy: MateInOnePolicy,
    dataset: Dataset,
    device: torch.device,
) -> float:
    if len(dataset) == 0:
        return 0.0

    puzzles = fens_from_dataset(dataset)
    env = ChessMateInOneEnv(puzzles=puzzles)
    result = evaluate_policy(
        policy=policy,
        env=env,
        episodes=len(puzzles),
        device=device,
    )
    return result.success_rate


def fens_from_dataset(dataset: Dataset) -> list[str]:
    if isinstance(dataset, Subset):
        return [dataset.dataset.puzzles[index].fen for index in dataset.indices]
    return [puzzle.fen for puzzle in dataset.puzzles]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-puzzles", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    _, result = train_supervised_policy(
        SupervisedTrainingConfig(
            puzzles_file=args.puzzles_file,
            validation_file=args.validation_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            train_fraction=args.train_fraction,
            seed=args.seed,
            device=args.device,
            max_puzzles=args.max_puzzles,
            log_every=args.log_every,
            save_path=args.save_path,
            load_path=args.load_path,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume_from,
        )
    )

    print()
    print("Supervised training summary")
    print(f"Train accuracy:          {result.train_accuracy.accuracy:.1%}")
    print(f"Validation accuracy:     {result.validation_accuracy.accuracy:.1%}")
    print(f"Validation mate success: {result.validation_mate_success_rate:.1%}")
    if args.save_path is not None:
        print(f"Saved policy:            {args.save_path}")


if __name__ == "__main__":
    main()
