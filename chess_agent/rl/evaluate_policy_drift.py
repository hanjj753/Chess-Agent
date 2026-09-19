import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import chess
import numpy as np
from sb3_contrib.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
import torch

from chess_agent.rl.actions import action_to_move, legal_action_mask, move_to_action
from chess_agent.rl.evaluate_full_chess_ppo import infer_history_length, save_report
from chess_agent.rl.evaluate_tactical import parse_episode_count
from chess_agent.rl.observations import boards_to_history_observation
from chess_agent.rl.tactical_puzzle_env import (
    TacticalPuzzle,
    load_tactical_puzzles,
)
from chess_agent.rl.train_full_chess_ppo import TrackedMaskablePPO


@dataclass(frozen=True)
class TacticalPolicyProbe:
    puzzle_index: int
    agent_move_index: int
    fen: str
    rating: int | None
    themes: tuple[str, ...]
    expected_move_uci: str
    observation: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True)
class PolicyDriftPosition:
    puzzle_index: int
    agent_move_index: int
    fen: str
    rating: int | None
    themes: tuple[str, ...]
    expected_move_uci: str
    legal_move_count: int
    reference_top_move_uci: str
    candidate_top_move_uci: str
    reference_entropy: float
    candidate_entropy: float
    reference_correct_probability: float
    candidate_correct_probability: float
    kl_reference_to_candidate: float
    js_divergence: float

    @property
    def top1_agreement(self) -> bool:
        return self.reference_top_move_uci == self.candidate_top_move_uci

    @property
    def reference_correct(self) -> bool:
        return self.reference_top_move_uci == self.expected_move_uci

    @property
    def candidate_correct(self) -> bool:
        return self.candidate_top_move_uci == self.expected_move_uci


@dataclass(frozen=True)
class PolicyDriftResult:
    puzzles: int
    history_length: int
    positions: tuple[PolicyDriftPosition, ...]

    @property
    def mean_reference_entropy(self) -> float:
        return mean(position.reference_entropy for position in self.positions)

    @property
    def mean_candidate_entropy(self) -> float:
        return mean(position.candidate_entropy for position in self.positions)

    @property
    def mean_kl_reference_to_candidate(self) -> float:
        return mean(position.kl_reference_to_candidate for position in self.positions)

    @property
    def mean_js_divergence(self) -> float:
        return mean(position.js_divergence for position in self.positions)

    @property
    def top1_agreement_rate(self) -> float:
        return fraction(position.top1_agreement for position in self.positions)

    @property
    def reference_accuracy(self) -> float:
        return fraction(position.reference_correct for position in self.positions)

    @property
    def candidate_accuracy(self) -> float:
        return fraction(position.candidate_correct for position in self.positions)

    @property
    def mean_reference_correct_probability(self) -> float:
        return mean(
            position.reference_correct_probability for position in self.positions
        )

    @property
    def mean_candidate_correct_probability(self) -> float:
        return mean(
            position.candidate_correct_probability for position in self.positions
        )

    @property
    def candidate_only_correct(self) -> int:
        return sum(
            position.candidate_correct and not position.reference_correct
            for position in self.positions
        )

    @property
    def reference_only_correct(self) -> int:
        return sum(
            position.reference_correct and not position.candidate_correct
            for position in self.positions
        )


def iter_tactical_policy_probes(
    puzzles: Sequence[TacticalPuzzle],
    *,
    history_length: int,
    max_puzzles: int | None = None,
) -> Iterator[TacticalPolicyProbe]:
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if max_puzzles is not None and max_puzzles < 1:
        raise ValueError("max_puzzles must be positive or None")

    selected = puzzles if max_puzzles is None else puzzles[:max_puzzles]
    for puzzle_index, puzzle in enumerate(selected):
        board = chess.Board(puzzle.initial_fen)
        board_history = [board.copy(stack=True)]
        agent_move_index = 0
        for line_index, move_uci in enumerate(puzzle.line_uci):
            move = chess.Move.from_uci(move_uci)
            if line_index % 2 == 0:
                yield TacticalPolicyProbe(
                    puzzle_index=puzzle_index,
                    agent_move_index=agent_move_index,
                    fen=board.fen(),
                    rating=puzzle.rating,
                    themes=puzzle.themes,
                    expected_move_uci=move_uci,
                    observation=boards_to_history_observation(
                        board_history,
                        history_length=history_length,
                    ),
                    action_mask=legal_action_mask(board).astype(bool),
                )
                agent_move_index += 1

            board.push(move)
            board_history.append(board.copy(stack=True))
            max_frames = history_length + 1
            if len(board_history) > max_frames:
                del board_history[:-max_frames]


@torch.no_grad()
def evaluate_policy_drift(
    *,
    reference_model: TrackedMaskablePPO,
    candidate_model: TrackedMaskablePPO,
    puzzles: Sequence[TacticalPuzzle],
    max_puzzles: int | None = None,
    batch_size: int = 256,
) -> PolicyDriftResult:
    if not puzzles:
        raise ValueError("at least one puzzle is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    reference_history = infer_history_length(reference_model)
    candidate_history = infer_history_length(candidate_model)
    if reference_history != candidate_history:
        raise ValueError(
            "reference and candidate models use different history lengths: "
            f"{reference_history} != {candidate_history}"
        )

    selected_puzzles = (
        len(puzzles) if max_puzzles is None else min(max_puzzles, len(puzzles))
    )
    probes = iter_tactical_policy_probes(
        puzzles,
        history_length=reference_history,
        max_puzzles=max_puzzles,
    )
    positions: list[PolicyDriftPosition] = []
    policies = tuple(
        dict.fromkeys((reference_model.policy, candidate_model.policy))
    )
    training_modes = tuple(policy.training for policy in policies)
    for policy in policies:
        policy.set_training_mode(False)

    try:
        for batch in batched(probes, batch_size):
            observations = np.stack([probe.observation for probe in batch])
            action_masks = np.stack([probe.action_mask for probe in batch])
            reference_probabilities = model_action_probabilities(
                reference_model,
                observations,
                action_masks,
            )
            candidate_probabilities = model_action_probabilities(
                candidate_model,
                observations,
                action_masks,
            )
            for probe, reference_probs, candidate_probs in zip(
                batch,
                reference_probabilities,
                candidate_probabilities,
                strict=True,
            ):
                positions.append(
                    compare_position_probabilities(
                        probe,
                        reference_probs=reference_probs,
                        candidate_probs=candidate_probs,
                    )
                )
    finally:
        for policy, was_training in zip(policies, training_modes, strict=True):
            policy.set_training_mode(was_training)

    if not positions:
        raise ValueError("no tactical agent positions were generated")
    return PolicyDriftResult(
        puzzles=selected_puzzles,
        history_length=reference_history,
        positions=tuple(positions),
    )


def model_action_probabilities(
    model: TrackedMaskablePPO,
    observations: np.ndarray,
    action_masks: np.ndarray,
) -> np.ndarray:
    observation_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=model.device,
    )
    distribution = model.policy.get_distribution(
        observation_tensor,
        action_masks=action_masks,
    )
    if not isinstance(distribution, MaskableCategoricalDistribution):
        raise TypeError("PPO model does not use a categorical action distribution")
    return distribution.distribution.probs.detach().cpu().numpy()


def compare_position_probabilities(
    probe: TacticalPolicyProbe,
    *,
    reference_probs: np.ndarray,
    candidate_probs: np.ndarray,
) -> PolicyDriftPosition:
    legal_actions = np.flatnonzero(probe.action_mask)
    if legal_actions.size == 0:
        raise ValueError(f"probe has no legal moves: {probe.fen}")

    reference = normalized_legal_probabilities(reference_probs, legal_actions)
    candidate = normalized_legal_probabilities(candidate_probs, legal_actions)
    expected_action = move_to_action(chess.Move.from_uci(probe.expected_move_uci))
    expected_matches = np.flatnonzero(legal_actions == expected_action)
    if expected_matches.size != 1:
        raise ValueError(
            f"expected move is not legal: {probe.expected_move_uci} from {probe.fen}"
        )
    expected_index = int(expected_matches[0])

    reference_top_action = int(legal_actions[int(np.argmax(reference))])
    candidate_top_action = int(legal_actions[int(np.argmax(candidate))])
    midpoint = 0.5 * (reference + candidate)
    return PolicyDriftPosition(
        puzzle_index=probe.puzzle_index,
        agent_move_index=probe.agent_move_index,
        fen=probe.fen,
        rating=probe.rating,
        themes=probe.themes,
        expected_move_uci=probe.expected_move_uci,
        legal_move_count=int(legal_actions.size),
        reference_top_move_uci=action_to_move(reference_top_action).uci(),
        candidate_top_move_uci=action_to_move(candidate_top_action).uci(),
        reference_entropy=entropy(reference),
        candidate_entropy=entropy(candidate),
        reference_correct_probability=float(reference[expected_index]),
        candidate_correct_probability=float(candidate[expected_index]),
        kl_reference_to_candidate=kl_divergence(reference, candidate),
        js_divergence=0.5
        * (
            kl_divergence(reference, midpoint)
            + kl_divergence(candidate, midpoint)
        ),
    )


def normalized_legal_probabilities(
    probabilities: np.ndarray,
    legal_actions: np.ndarray,
) -> np.ndarray:
    legal = np.asarray(probabilities, dtype=np.float64)[legal_actions]
    total = float(np.sum(legal))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("policy probabilities are not a positive finite distribution")
    return legal / total


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).tiny
    safe_left = np.clip(left, epsilon, 1.0)
    safe_right = np.clip(right, epsilon, 1.0)
    return float(np.sum(safe_left * (np.log(safe_left) - np.log(safe_right))))


def format_policy_drift_report(
    *,
    reference_model_path: str | Path,
    candidate_model_path: str | Path,
    puzzles_file: str | Path,
    result: PolicyDriftResult,
) -> str:
    entropy_delta = result.mean_candidate_entropy - result.mean_reference_entropy
    probability_delta = (
        result.mean_candidate_correct_probability
        - result.mean_reference_correct_probability
    )
    accuracy_delta = result.candidate_accuracy - result.reference_accuracy
    return "\n".join(
        (
            "PPO policy drift evaluation",
            f"Reference:       {reference_model_path}",
            f"Candidate:       {candidate_model_path}",
            f"Puzzles file:    {puzzles_file}",
            f"Puzzles:         {result.puzzles}",
            f"Agent positions: {len(result.positions)}",
            f"History length:  {result.history_length}",
            "",
            "Distribution drift",
            f"Reference entropy:         {result.mean_reference_entropy:.4f}",
            f"Candidate entropy:         {result.mean_candidate_entropy:.4f}",
            f"Entropy delta:             {entropy_delta:+.4f}",
            f"Reference effective moves: {math.exp(result.mean_reference_entropy):.2f}",
            f"Candidate effective moves: {math.exp(result.mean_candidate_entropy):.2f}",
            f"KL(reference || candidate): {result.mean_kl_reference_to_candidate:.6f}",
            f"Jensen-Shannon divergence:  {result.mean_js_divergence:.6f}",
            f"Top-1 agreement:           {result.top1_agreement_rate:.1%}",
            "",
            "Tactical move retention",
            f"Reference top-1 accuracy:  {result.reference_accuracy:.1%}",
            f"Candidate top-1 accuracy:  {result.candidate_accuracy:.1%}",
            f"Accuracy delta:            {accuracy_delta:+.1%}",
            f"Reference correct prob:    {result.mean_reference_correct_probability:.1%}",
            f"Candidate correct prob:    {result.mean_candidate_correct_probability:.1%}",
            f"Correct-prob delta:        {probability_delta:+.1%}",
            f"Candidate-only correct:    {result.candidate_only_correct}",
            f"Reference-only correct:    {result.reference_only_correct}",
            "",
            "Interpretation",
            "Positive entropy delta means the candidate distributes probability",
            "across more legal moves. KL/JS near zero and high top-1 agreement mean",
            "the candidate stayed close to the pretrained reference policy.",
        )
    ) + "\n"


def save_positions_csv(
    path: str | Path,
    *,
    result: PolicyDriftResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "puzzle_index",
                "agent_move_index",
                "fen",
                "rating",
                "themes",
                "expected_move_uci",
                "legal_move_count",
                "reference_top_move_uci",
                "candidate_top_move_uci",
                "top1_agreement",
                "reference_correct",
                "candidate_correct",
                "reference_entropy",
                "candidate_entropy",
                "entropy_delta",
                "reference_correct_probability",
                "candidate_correct_probability",
                "correct_probability_delta",
                "kl_reference_to_candidate",
                "js_divergence",
            ),
        )
        writer.writeheader()
        for position in result.positions:
            writer.writerow(
                {
                    "puzzle_index": position.puzzle_index,
                    "agent_move_index": position.agent_move_index,
                    "fen": position.fen,
                    "rating": position.rating,
                    "themes": " ".join(position.themes),
                    "expected_move_uci": position.expected_move_uci,
                    "legal_move_count": position.legal_move_count,
                    "reference_top_move_uci": position.reference_top_move_uci,
                    "candidate_top_move_uci": position.candidate_top_move_uci,
                    "top1_agreement": int(position.top1_agreement),
                    "reference_correct": int(position.reference_correct),
                    "candidate_correct": int(position.candidate_correct),
                    "reference_entropy": position.reference_entropy,
                    "candidate_entropy": position.candidate_entropy,
                    "entropy_delta": (
                        position.candidate_entropy - position.reference_entropy
                    ),
                    "reference_correct_probability": (
                        position.reference_correct_probability
                    ),
                    "candidate_correct_probability": (
                        position.candidate_correct_probability
                    ),
                    "correct_probability_delta": (
                        position.candidate_correct_probability
                        - position.reference_correct_probability
                    ),
                    "kl_reference_to_candidate": (
                        position.kl_reference_to_candidate
                    ),
                    "js_divergence": position.js_divergence,
                }
            )
    return output_path


def default_positions_output_path(report_path: str | Path) -> Path:
    path = Path(report_path)
    return path.with_name(f"{path.stem}_positions.csv")


def batched(
    values: Iterable[TacticalPolicyProbe],
    batch_size: int,
) -> Iterator[tuple[TacticalPolicyProbe, ...]]:
    batch: list[TacticalPolicyProbe] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def fraction(values: Iterable[bool]) -> float:
    collected = tuple(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model-path", type=Path, required=True)
    parser.add_argument("--candidate-model-path", type=Path, required=True)
    parser.add_argument("--puzzles-file", type=Path, required=True)
    parser.add_argument(
        "--puzzles",
        type=parse_episode_count,
        default=1_000,
        help="number of puzzles to probe, or 'all'",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--positions-output-path", type=Path)
    args = parser.parse_args()

    puzzles = load_tactical_puzzles(args.puzzles_file)
    reference_model = TrackedMaskablePPO.load(
        args.reference_model_path,
        device=args.device,
    )
    candidate_model = TrackedMaskablePPO.load(
        args.candidate_model_path,
        device=args.device,
    )
    result = evaluate_policy_drift(
        reference_model=reference_model,
        candidate_model=candidate_model,
        puzzles=puzzles,
        max_puzzles=args.puzzles,
        batch_size=args.batch_size,
    )
    report = format_policy_drift_report(
        reference_model_path=args.reference_model_path,
        candidate_model_path=args.candidate_model_path,
        puzzles_file=args.puzzles_file,
        result=result,
    )
    print(report, end="")
    if args.output_path is not None:
        saved_path = save_report(args.output_path, report)
        print(f"Saved report:    {saved_path}")

    positions_output_path = args.positions_output_path
    if positions_output_path is None and args.output_path is not None:
        positions_output_path = default_positions_output_path(args.output_path)
    if positions_output_path is not None:
        saved_positions = save_positions_csv(
            positions_output_path,
            result=result,
        )
        print(f"Saved positions: {saved_positions}")


if __name__ == "__main__":
    main()
