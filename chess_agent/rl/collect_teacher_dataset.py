import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from chess_agent.agents.uci_engine_agent import resolve_engine_path
from chess_agent.rl.actions import action_to_move, move_to_action
from chess_agent.rl.collect_value_dataset import (
    infer_history_length,
    load_collection_policy,
    make_validation_indices,
    model_observation_shape,
    resolve_device,
    select_policy_action,
)
from chess_agent.rl.experiment_tracking import to_jsonable
from chess_agent.rl.full_chess_env import BoardOnlyObservation, FullChessEnv
from chess_agent.rl.teacher_dataset import (
    COLUMN_DTYPES,
    TeacherDataset,
    empty_teacher_columns,
    load_teacher_dataset,
    save_teacher_dataset,
)
from chess_agent.rl.train_full_chess_ppo import make_opponent
from chess_agent.rl.value_dataset import pack_observations


@dataclass(frozen=True)
class TeacherCollectionConfig:
    model_path: Path
    output_dir: Path
    engine_path: Path = Path("stockfish")
    games: int = 500
    positions_per_game: int = 40
    skip_plies: int = 8
    validation_fraction: float = 0.1
    opponent: str = "alpha-random"
    alpha_move_probability: float = 0.5
    opponent_depth: int = 1
    max_plies: int = 200
    deterministic_policy: bool = False
    temperature: float = 1.0
    engine_time: float | None = None
    engine_depth: int | None = None
    engine_nodes: int | None = None
    engine_threads: int = 1
    engine_hash: int = 128
    seed: int = 410000
    device: str = "auto"
    log_every: int = 5
    resume: bool = False
    summary_output: Path | None = None


@dataclass(frozen=True)
class PositionSample:
    board: chess.Board
    observation: np.ndarray
    action_mask: np.ndarray
    played_action: int
    ply: int


def engine_limit(config: TeacherCollectionConfig) -> chess.engine.Limit:
    return chess.engine.Limit(
        time=(
            0.05 if all(value is None for value in (
                config.engine_time, config.engine_depth, config.engine_nodes,
            )) else config.engine_time
        ),
        depth=config.engine_depth,
        nodes=config.engine_nodes,
    )


def score_columns(info: dict[str, Any], perspective: chess.Color) -> tuple[float, float]:
    score = info.get("score")
    if score is None:
        raise ValueError("teacher returned no evaluation score")
    relative = score.pov(perspective)
    if relative.is_mate():
        mate = relative.mate()
        if mate is not None:
            return math.nan, float(mate)
    cp = relative.score()
    if cp is None:
        raise ValueError("teacher returned an unusable evaluation score")
    return float(cp), math.nan


def label_position(
    engine: chess.engine.SimpleEngine,
    sample: PositionSample,
    limit: chess.engine.Limit,
) -> tuple[int, float, float, float, float]:
    # A fresh engine game per sample avoids TT state from earlier games/resumes.
    game = object()
    best = engine.analyse(sample.board, limit, game=game)
    pv = best.get("pv", [])
    if not pv or pv[0] not in sample.board.legal_moves:
        raise ValueError(f"teacher returned no legal principal variation: {sample.board.fen()}")
    teacher_action = move_to_action(pv[0])
    teacher_cp, teacher_mate = score_columns(best, sample.board.turn)
    if teacher_action == sample.played_action:
        played_cp, played_mate = teacher_cp, teacher_mate
    else:
        played_move = action_to_move(sample.played_action)
        if played_move not in sample.board.legal_moves:
            raise ValueError("collection policy selected an illegal action")
        played = engine.analyse(
            sample.board, limit, game=game, root_moves=[played_move],
        )
        played_cp, played_mate = score_columns(played, sample.board.turn)
    return teacher_action, teacher_cp, teacher_mate, played_cp, played_mate


def label_game(
    *,
    samples: list[PositionSample],
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    game_id: int,
    observation_shape: tuple[int, int, int],
    metadata: dict[str, Any],
) -> TeacherDataset:
    columns = empty_teacher_columns(observation_shape)
    if samples:
        labels = [label_position(engine, sample, limit) for sample in samples]
        columns.update({
            "packed_observations": pack_observations(np.stack([s.observation for s in samples])),
            "packed_action_masks": np.packbits(
                np.stack([s.action_mask for s in samples]), axis=1, bitorder="little",
            ),
            "teacher_actions": np.asarray([row[0] for row in labels], dtype=np.int32),
            "played_actions": np.asarray([s.played_action for s in samples], dtype=np.int32),
            "teacher_cp": np.asarray([row[1] for row in labels], dtype=np.float32),
            "teacher_mate": np.asarray([row[2] for row in labels], dtype=np.float32),
            "played_cp": np.asarray([row[3] for row in labels], dtype=np.float32),
            "played_mate": np.asarray([row[4] for row in labels], dtype=np.float32),
            "game_ids": np.full(len(samples), game_id, dtype=np.int32),
            "plies": np.asarray([s.ply for s in samples], dtype=np.int32),
            "fens": np.asarray([s.board.fen() for s in samples]),
        })
    return TeacherDataset(columns, observation_shape, metadata)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_manifest(config: TeacherCollectionConfig, engine_path: Path) -> dict[str, Any]:
    raw_config = to_jsonable(config)
    for key in ("resume", "log_every", "summary_output", "output_dir"):
        raw_config.pop(key)
    raw_config["model_path"] = str(config.model_path.resolve())
    raw_config["engine_path"] = str(engine_path.resolve())
    manifest = {
        "version": 1,
        "config": raw_config,
        "model_sha256": file_sha256(config.model_path),
        "engine_sha256": file_sha256(engine_path),
    }
    destination = config.output_dir / "manifest.json"
    if config.resume:
        if not destination.is_file():
            raise ValueError("--resume requires an existing manifest.json")
        if json.loads(destination.read_text(encoding="utf-8")) != manifest:
            raise ValueError("resume configuration/model/engine differs from the saved manifest")
    else:
        if config.output_dir.exists() and any(config.output_dir.iterdir()):
            raise FileExistsError("output directory is not empty; use --resume or a new directory")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def collect_teacher_dataset(config: TeacherCollectionConfig) -> dict[str, Any]:
    validate_config(config)
    engine_path = resolve_engine_path(config.engine_path)
    manifest = prepare_manifest(config, engine_path)
    validation_ids = make_validation_indices(
        games=config.games,
        validation_fraction=config.validation_fraction,
        rng=np.random.default_rng(config.seed),
    )
    missing = [
        game_id for game_id in range(1, config.games + 1)
        if not shard_path(config, game_id).is_file()
    ]
    if missing:
        device = resolve_device(config.device)
        model = load_collection_policy(config.model_path, device=device)
        shape = model_observation_shape(model)
        env = BoardOnlyObservation(FullChessEnv(
            opponent=make_opponent(
                config.opponent, alpha_move_probability=config.alpha_move_probability,
                depth=config.opponent_depth, time_limit=None,
            ),
            history_length=infer_history_length(model), max_plies=config.max_plies,
        ))
        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
            options = {
                name: value for name, value in {
                    "Threads": config.engine_threads,
                    "Hash": config.engine_hash,
                    "UCI_LimitStrength": False,
                }.items() if name in engine.options
            }
            engine.configure(options)
            print(
                f"Teacher={engine.id.get('name', engine_path.name)} "
                f"limit={engine_limit(config)} cached_games={config.games - len(missing)} "
                f"remaining_games={len(missing)}",
                flush=True,
            )
            for index, game_id in enumerate(missing, start=1):
                # Per-game RNGs make skipped cached games independent of new games.
                game_seed = config.seed + game_id - 1
                torch.manual_seed(game_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(game_seed)
                color = chess.WHITE if game_id % 2 else chess.BLACK
                observation, _ = env.reset(seed=game_seed, options={"agent_color": color})
                trajectory: list[PositionSample] = []
                terminated = truncated = False
                while not (terminated or truncated):
                    mask = env.action_masks()
                    action = select_policy_action(
                        model=model, observation=observation, action_mask=mask,
                        device=device, deterministic=config.deterministic_policy,
                        temperature=config.temperature,
                    )
                    board = env.unwrapped.board
                    assert board is not None
                    if env.unwrapped.episode_plies >= config.skip_plies:
                        trajectory.append(PositionSample(
                            board.copy(stack=True), np.asarray(observation).copy(),
                            mask.copy(), action, env.unwrapped.episode_plies,
                        ))
                    observation, _, terminated, truncated, info = env.step(action)
                rng = np.random.default_rng(game_seed)
                indices = sorted(rng.choice(
                    len(trajectory), size=min(config.positions_per_game, len(trajectory)),
                    replace=False,
                ).tolist())
                selected = [trajectory[position] for position in indices]
                final_board = env.unwrapped.board
                assert final_board is not None
                pgn = chess.pgn.Game.from_board(final_board)
                pgn.headers.update({
                    "Event": "Teacher dataset collection",
                    "White": "policy" if color else config.opponent,
                    "Black": config.opponent if color else "policy",
                    "Result": str(info["result"]),
                    "GameId": str(game_id), "Seed": str(game_seed),
                })
                dataset = label_game(
                    samples=selected, engine=engine, limit=engine_limit(config),
                    game_id=game_id, observation_shape=shape,
                    metadata={
                        "game_id": game_id, "seed": game_seed,
                        "split": "validation" if game_id in validation_ids else "train",
                        "agent_color": "white" if color else "black",
                        "result": str(info["result"]),
                        "termination": str(info["termination"]),
                        "game_plies": env.unwrapped.episode_plies,
                        "teacher_name": engine.id.get("name", engine_path.name),
                        "teacher_options": options,
                        "pgn": str(pgn),
                    },
                )
                save_teacher_dataset(shard_path(config, game_id), dataset)
                if config.log_every and (index == 1 or index % config.log_every == 0 or index == len(missing)):
                    print(
                        f"game={game_id}/{config.games} labeled={len(dataset)} "
                        f"split={dataset.metadata['split']} "
                        f"completed={config.games - len(missing) + index}/{config.games}",
                        flush=True,
                    )
        finally:
            try:
                if engine is not None:
                    engine.quit()
            finally:
                env.close()
    return merge_teacher_games(config, manifest, validation_ids)


def shard_path(config: TeacherCollectionConfig, game_id: int) -> Path:
    return config.output_dir / "games" / f"game_{game_id:06d}.npz"


def merge_teacher_games(
    config: TeacherCollectionConfig,
    manifest: dict[str, Any],
    validation_ids: set[int],
) -> dict[str, Any]:
    splits: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "validation": []}
    seen: set[bytes] = set()
    duplicates = 0
    shape = None
    outcomes: dict[str, int] = {}
    shards: dict[int, dict[str, Any]] = {}
    # Give validation ownership of shared inputs, then remove them from train.
    game_ids = sorted(range(1, config.games + 1), key=lambda g: (g not in validation_ids, g))
    for game_id in game_ids:
        dataset = load_teacher_dataset(shard_path(config, game_id))
        if shape is not None and shape != dataset.observation_shape:
            raise ValueError("cached games have inconsistent observation shapes")
        shape = dataset.observation_shape
        split = "validation" if game_id in validation_ids else "train"
        if (
            dataset.metadata.get("game_id") != game_id
            or dataset.metadata.get("split") != split
            or not np.all(dataset.columns["game_ids"] == game_id)
        ):
            raise ValueError("cached game metadata does not match the manifest split")
        shards[game_id] = dataset.metadata
        result = dataset.metadata["result"]
        outcomes[result] = outcomes.get(result, 0) + 1
        keep = []
        for row, packed in enumerate(dataset.columns["packed_observations"]):
            key = packed.tobytes()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            keep.append(row)
        splits[split].append({name: values[keep] for name, values in dataset.columns.items()})
    assert shape is not None
    summary: dict[str, Any] = {
        "games": config.games, "duplicates_removed": duplicates,
        "model_path": str(config.model_path), "output_dir": str(config.output_dir),
        "config": manifest["config"],
        "teacher_name": shards[1]["teacher_name"],
        "results_white_pov": outcomes,
        "seed": config.seed, "observation_shape": list(shape),
    }
    csv_path = config.output_dir / "positions.csv"
    fields = (
        "split", "game_id", "ply", "agent_color", "fen", "played_uci", "teacher_uci",
        "teacher_cp", "played_cp", "teacher_mate", "played_mate", "cp_loss",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for split, batches in splits.items():
            columns = {
                name: np.concatenate([batch[name] for batch in batches])
                for name in COLUMN_DTYPES
            }
            dataset = TeacherDataset(columns, shape, {
                "split": split, "manifest": manifest,
                "score_perspective": "agent_to_move", "deduplication": "exact_observation",
            })
            if not len(dataset):
                raise ValueError(
                    f"{split} has no unique positions; collect more/longer games or reduce skip_plies"
                )
            path = save_teacher_dataset(config.output_dir / f"{split}.npz", dataset)
            c = dataset.columns
            cp_loss = dataset.cp_losses
            finite = np.isfinite(cp_loss)
            summary[split] = {
                "path": str(path), "positions": len(dataset),
                "assigned_games": config.games - len(validation_ids) if split == "train" else len(validation_ids),
                "games_with_samples": int(np.unique(c["game_ids"]).size),
                "teacher_agreement": float(np.mean(c["teacher_actions"] == c["played_actions"])),
                "cp_positions": int(np.count_nonzero(finite)),
                "mate_positions": int(np.count_nonzero(~finite)),
                "mean_cp_loss": float(np.mean(cp_loss[finite])) if np.any(finite) else None,
                "negative_raw_cp_differences": int(np.count_nonzero(c["teacher_cp"] < c["played_cp"])),
            }
            for row in range(len(dataset)):
                game_id = int(c["game_ids"][row])
                writer.writerow({
                    "split": split, "game_id": game_id, "ply": int(c["plies"][row]),
                    "agent_color": shards[game_id]["agent_color"], "fen": c["fens"][row],
                    "played_uci": action_to_move(int(c["played_actions"][row])).uci(),
                    "teacher_uci": action_to_move(int(c["teacher_actions"][row])).uci(),
                    **{
                        name: float(c[name][row]) if np.isfinite(c[name][row]) else ""
                        for name in ("teacher_cp", "played_cp", "teacher_mate", "played_mate")
                    },
                    "cp_loss": float(cp_loss[row]) if finite[row] else "",
                })
    (config.output_dir / "games.pgn").write_text(
        "\n\n".join(shards[g]["pgn"] for g in sorted(shards)) + "\n", encoding="utf-8",
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    report = build_summary(summary)
    (config.output_dir / "summary.txt").write_text(report, encoding="utf-8")
    if config.summary_output is not None:
        config.summary_output.parent.mkdir(parents=True, exist_ok=True)
        config.summary_output.write_text(report, encoding="utf-8")
    print(report, flush=True)
    return summary


def build_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Stockfish teacher dataset collection", "=" * 35,
        f"Source policy: {summary['model_path']}",
        f"Teacher:       {summary['teacher_name']}",
        f"Output:        {summary['output_dir']}",
        f"Games:         {summary['games']}",
        f"Seed:          {summary['seed']}",
        f"Opponent:      {summary['config']['opponent']} "
        f"(alpha probability={summary['config']['alpha_move_probability']:.1%})",
        f"Collection policy: {'deterministic' if summary['config']['deterministic_policy'] else 'stochastic'} "
        f"(temperature={summary['config']['temperature']:g})",
        f"Engine budget: time={summary['config']['engine_time']}, "
        f"depth={summary['config']['engine_depth']}, nodes={summary['config']['engine_nodes']} "
        "(0.05s if all are None)",
        f"Input shape:   {tuple(summary['observation_shape'])}",
        f"Duplicates removed: {summary['duplicates_removed']}",
        "", "Split         Games  Positions  Agreement  Avg cp loss  Mate rows",
    ]
    for split in ("train", "validation"):
        stats = summary[split]
        loss = "n/a" if stats['mean_cp_loss'] is None else f"{stats['mean_cp_loss']:.1f}"
        lines.append(
            f"{split:12s} {stats['games_with_samples']:5d} {stats['positions']:10d} "
            f"{stats['teacher_agreement']:9.1%} {loss:>12s} {stats['mate_positions']:10d}"
        )
        lines.append(
            f"  assigned_games={stats['assigned_games']} "
            f"cp_rows={stats['cp_positions']} "
            f"negative_raw_cp_differences={stats['negative_raw_cp_differences']}"
        )
    lines.extend([
        "", "Scores are from the agent-to-move perspective; mate is not converted to cp.",
        "Avg cp loss excludes mate rows; this is sampled-position loss, not playing Elo.",
        "Short searches can disagree: negative raw cp differences are clamped to 0.",
        "Validation is split by game; identical NN inputs are kept only once (validation first).",
        "Data collection only: no policy or value weights were updated.",
    ])
    return "\n".join(lines) + "\n"


def validate_config(config: TeacherCollectionConfig) -> None:
    if config.games < 2:
        raise ValueError("games must be at least 2")
    if config.positions_per_game < 1 or config.max_plies < 2:
        raise ValueError("positions_per_game must be positive and max_plies at least 2")
    if config.skip_plies < 0 or config.skip_plies >= config.max_plies:
        raise ValueError("skip_plies must be in [0, max_plies)")
    if not math.isfinite(config.validation_fraction) or not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.opponent not in ("random", "alpha-random", "alpha"):
        raise ValueError("unsupported collection opponent")
    if not math.isfinite(config.alpha_move_probability) or not 0 <= config.alpha_move_probability <= 1:
        raise ValueError("alpha_move_probability must be in [0, 1]")
    if config.opponent_depth < 1:
        raise ValueError("opponent_depth must be positive")
    if not math.isfinite(config.temperature) or config.temperature <= 0:
        raise ValueError("temperature must be positive")
    limits = (config.engine_time, config.engine_depth, config.engine_nodes)
    if sum(value is not None for value in limits) > 1:
        raise ValueError("choose only one engine limit: time, depth, or nodes")
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in limits):
        raise ValueError("engine limits must be finite and positive")
    if config.engine_threads < 1 or config.engine_hash < 1:
        raise ValueError("engine_threads and engine_hash must be positive")
    if config.seed < 0 or config.log_every < 0:
        raise ValueError("seed and log_every must be non-negative")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect on-policy positions with UCI teacher labels.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("stockfish"), dest="engine_path")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--positions-per-game", type=int, default=40)
    parser.add_argument("--skip-plies", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--opponent", choices=("random", "alpha-random", "alpha"), default="alpha-random")
    parser.add_argument("--alpha-move-probability", type=float, default=0.5)
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--engine-time", type=float)
    limits.add_argument("--engine-depth", type=int)
    limits.add_argument("--engine-nodes", type=int)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--engine-hash", type=int, default=128)
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    collect_teacher_dataset(TeacherCollectionConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
