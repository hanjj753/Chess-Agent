import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_agent.rl.full_chess_env import BoardOnlyObservation, FullChessEnv
from chess_agent.rl.observations import OBSERVATION_CHANNELS
from chess_agent.rl.policy_value import PolicyValueNetwork, load_policy_value
from chess_agent.rl.train_full_chess_ppo import make_opponent
from chess_agent.rl.value_dataset import pack_observations, save_value_dataset


VALUE_DATASET_OPPONENTS = ("random", "alpha", "mixed")


@dataclass
class SplitAccumulator:
    packed_observations: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    outcomes: list[np.ndarray] = field(default_factory=list)
    game_ids: list[np.ndarray] = field(default_factory=list)
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def positions(self) -> int:
        return sum(int(values.shape[0]) for values in self.targets)

    def add_game(
        self,
        *,
        observations: list[np.ndarray],
        reward: float,
        gamma: float,
        game_id: int,
    ) -> None:
        if not observations:
            return
        observation_array = np.stack(observations).astype(np.int8, copy=False)
        steps = observation_array.shape[0]
        discounts = np.power(
            gamma,
            np.arange(steps - 1, -1, -1, dtype=np.float32),
        )
        target_values = np.asarray(reward * discounts, dtype=np.float32)
        outcome = int(np.sign(reward))

        self.packed_observations.append(pack_observations(observation_array))
        self.targets.append(target_values)
        self.outcomes.append(np.full(steps, outcome, dtype=np.int8))
        self.game_ids.append(np.full(steps, game_id, dtype=np.int32))
        self.games += 1
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        else:
            self.draws += 1

    def arrays(self, *, observation_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
        if not self.packed_observations:
            packed_width = (int(np.prod(observation_shape)) + 7) // 8
            return {
                "packed_observations": np.empty((0, packed_width), dtype=np.uint8),
                "targets": np.empty(0, dtype=np.float32),
                "outcomes": np.empty(0, dtype=np.int8),
                "game_ids": np.empty(0, dtype=np.int32),
            }
        return {
            "packed_observations": np.concatenate(self.packed_observations),
            "targets": np.concatenate(self.targets),
            "outcomes": np.concatenate(self.outcomes),
            "game_ids": np.concatenate(self.game_ids),
        }


@dataclass(frozen=True)
class ValueDatasetCollectionResult:
    train_path: Path
    validation_path: Path
    train_games: int
    validation_games: int
    train_positions: int
    validation_positions: int
    wins: int
    draws: int
    losses: int


def collect_value_dataset(
    *,
    model_path: str | Path,
    train_output_path: str | Path,
    validation_output_path: str | Path,
    games: int,
    validation_fraction: float,
    opponent: str,
    alpha_fraction: float,
    opponent_depth: int,
    opponent_time_limit: float | None,
    max_plies: int,
    gamma: float,
    deterministic_policy: bool,
    temperature: float,
    seed: int,
    device: str,
    log_every: int,
) -> ValueDatasetCollectionResult:
    validate_collection_options(
        games=games,
        validation_fraction=validation_fraction,
        opponent=opponent,
        alpha_fraction=alpha_fraction,
        max_plies=max_plies,
        gamma=gamma,
        temperature=temperature,
        log_every=log_every,
    )
    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    model = load_policy_value(model_path, device=resolved_device)
    history_length = infer_history_length(model)
    observation_shape = (model.input_channels, 8, 8)
    opponent_schedule = make_opponent_schedule(
        games=games,
        opponent=opponent,
        alpha_fraction=alpha_fraction,
        rng=rng,
    )
    validation_indices = make_validation_indices(
        games=games,
        validation_fraction=validation_fraction,
        rng=rng,
    )
    opponent_kinds = tuple(sorted(set(opponent_schedule)))
    environments = {
        kind: BoardOnlyObservation(
            FullChessEnv(
                opponent=make_opponent(
                    kind,
                    depth=opponent_depth,
                    time_limit=opponent_time_limit,
                ),
                history_length=history_length,
                max_plies=max_plies,
            )
        )
        for kind in opponent_kinds
    }
    train = SplitAccumulator()
    validation = SplitAccumulator()

    try:
        for episode in range(1, games + 1):
            opponent_kind = opponent_schedule[episode - 1]
            env = environments[opponent_kind]
            agent_color = chess.WHITE if episode % 2 == 1 else chess.BLACK
            observation, _ = env.reset(
                seed=seed + episode - 1,
                options={"agent_color": agent_color},
            )
            trajectory: list[np.ndarray] = []
            episode_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                trajectory.append(np.asarray(observation, dtype=np.int8).copy())
                action = select_policy_action(
                    model=model,
                    observation=observation,
                    action_mask=env.action_masks(),
                    device=resolved_device,
                    deterministic=deterministic_policy,
                    temperature=temperature,
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

            target_split = validation if episode in validation_indices else train
            target_split.add_game(
                observations=trajectory,
                reward=episode_reward,
                gamma=gamma,
                game_id=episode,
            )
            if log_every and (episode % log_every == 0 or episode == games):
                print(
                    f"game={episode:6d}/{games} "
                    f"positions={train.positions + validation.positions:8d} "
                    f"W/D/L={train.wins + validation.wins}/"
                    f"{train.draws + validation.draws}/"
                    f"{train.losses + validation.losses}",
                    flush=True,
                )
    finally:
        for env in environments.values():
            env.close()

    common_metadata: dict[str, Any] = {
        "source_model": str(model_path),
        "games": games,
        "validation_fraction": validation_fraction,
        "opponent": opponent,
        "alpha_fraction": alpha_fraction if opponent == "mixed" else None,
        "opponent_depth": opponent_depth,
        "opponent_time_limit": opponent_time_limit,
        "max_plies": max_plies,
        "gamma": gamma,
        "deterministic_policy": deterministic_policy,
        "temperature": temperature,
        "seed": seed,
        "history_length": history_length,
    }
    train_arrays = train.arrays(observation_shape=observation_shape)
    validation_arrays = validation.arrays(observation_shape=observation_shape)
    train_path = save_value_dataset(
        train_output_path,
        **train_arrays,
        observation_shape=observation_shape,
        metadata={**common_metadata, "split": "train"},
    )
    validation_path = save_value_dataset(
        validation_output_path,
        **validation_arrays,
        observation_shape=observation_shape,
        metadata={**common_metadata, "split": "validation"},
    )
    return ValueDatasetCollectionResult(
        train_path=train_path,
        validation_path=validation_path,
        train_games=train.games,
        validation_games=validation.games,
        train_positions=train.positions,
        validation_positions=validation.positions,
        wins=train.wins + validation.wins,
        draws=train.draws + validation.draws,
        losses=train.losses + validation.losses,
    )


@torch.no_grad()
def select_policy_action(
    *,
    model: PolicyValueNetwork,
    observation: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
    deterministic: bool,
    temperature: float,
) -> int:
    board = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    legal_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    logits, _ = model(board)
    masked_logits = (logits / temperature).masked_fill(~legal_mask, -torch.inf)
    if deterministic:
        return int(torch.argmax(masked_logits, dim=-1).item())
    probabilities = torch.softmax(masked_logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def infer_history_length(model: PolicyValueNetwork) -> int:
    if model.input_channels % OBSERVATION_CHANNELS != 0:
        raise ValueError("policy-value input channels must be a multiple of 18")
    return model.input_channels // OBSERVATION_CHANNELS - 1


def make_opponent_schedule(
    *,
    games: int,
    opponent: str,
    alpha_fraction: float,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    if opponent != "mixed":
        return (opponent,) * games
    alpha_games = int(round(games * alpha_fraction))
    schedule = np.asarray(
        ["alpha"] * alpha_games + ["random"] * (games - alpha_games),
        dtype=object,
    )
    rng.shuffle(schedule)
    return tuple(str(value) for value in schedule)


def make_validation_indices(
    *,
    games: int,
    validation_fraction: float,
    rng: np.random.Generator,
) -> set[int]:
    validation_games = max(1, min(games - 1, int(round(games * validation_fraction))))
    indices = rng.choice(
        np.arange(1, games + 1),
        size=validation_games,
        replace=False,
    )
    return {int(value) for value in indices}


def validate_collection_options(
    *,
    games: int,
    validation_fraction: float,
    opponent: str,
    alpha_fraction: float,
    max_plies: int,
    gamma: float,
    temperature: float,
    log_every: int,
) -> None:
    if games < 2:
        raise ValueError("games must be at least 2 for a train/validation split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if opponent not in VALUE_DATASET_OPPONENTS:
        raise ValueError(f"unsupported opponent: {opponent}")
    if not 0 <= alpha_fraction <= 1:
        raise ValueError("alpha_fraction must be in [0, 1]")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if log_every < 0:
        raise ValueError("log_every must be non-negative")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--opponent", choices=VALUE_DATASET_OPPONENTS, default="mixed")
    parser.add_argument("--alpha-fraction", type=float, default=0.5)
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--opponent-time-limit", type=float)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    result = collect_value_dataset(
        model_path=args.model_path,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        games=args.games,
        validation_fraction=args.validation_fraction,
        opponent=args.opponent,
        alpha_fraction=args.alpha_fraction,
        opponent_depth=args.opponent_depth,
        opponent_time_limit=args.opponent_time_limit,
        max_plies=args.max_plies,
        gamma=args.gamma,
        deterministic_policy=args.deterministic_policy,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )
    print()
    print("Full-chess value dataset summary")
    print(f"Games:               {result.train_games + result.validation_games}")
    print(f"W/D/L:               {result.wins}/{result.draws}/{result.losses}")
    print(f"Train games:         {result.train_games}")
    print(f"Validation games:    {result.validation_games}")
    print(f"Train positions:     {result.train_positions}")
    print(f"Validation positions:{result.validation_positions:>8d}")
    print(f"Train dataset:       {result.train_path}")
    print(f"Validation dataset:  {result.validation_path}")


if __name__ == "__main__":
    main()
