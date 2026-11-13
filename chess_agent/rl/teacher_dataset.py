from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from chess_agent.rl.actions import ACTION_SIZE


TEACHER_DATASET_VERSION = 1
COLUMN_DTYPES = {
    "packed_observations": np.uint8,
    "packed_action_masks": np.uint8,
    "teacher_actions": np.int32,
    "played_actions": np.int32,
    "teacher_cp": np.float32,
    "played_cp": np.float32,
    "teacher_mate": np.float32,
    "played_mate": np.float32,
    "game_ids": np.int32,
    "plies": np.int32,
    "fens": np.str_,
}


@dataclass(frozen=True)
class TeacherDataset:
    columns: dict[str, np.ndarray]
    observation_shape: tuple[int, int, int]
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return int(self.columns["teacher_actions"].size)

    def observations(self, indices: np.ndarray) -> np.ndarray:
        unpacked = np.unpackbits(
            self.columns["packed_observations"][indices],
            axis=1,
            count=int(np.prod(self.observation_shape)),
            bitorder="little",
        )
        return unpacked.reshape((-1, *self.observation_shape)).astype(np.float32)

    def action_masks(self, indices: np.ndarray) -> np.ndarray:
        return np.unpackbits(
            self.columns["packed_action_masks"][indices],
            axis=1,
            count=ACTION_SIZE,
            bitorder="little",
        ).astype(bool)

    @property
    def cp_losses(self) -> np.ndarray:
        return np.maximum(0, self.columns["teacher_cp"] - self.columns["played_cp"])


def empty_teacher_columns(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        name: np.empty(_column_shape(name, 0, shape), dtype=dtype)
        for name, dtype in COLUMN_DTYPES.items()
    }


def _column_shape(
    name: str, count: int, observation_shape: tuple[int, int, int],
) -> tuple[int, ...]:
    if name == "packed_observations":
        return count, (int(np.prod(observation_shape)) + 7) // 8
    if name == "packed_action_masks":
        return count, (ACTION_SIZE + 7) // 8
    return (count,)


def save_teacher_dataset(path: str | Path, dataset: TeacherDataset) -> Path:
    validate_teacher_dataset(dataset)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **dataset.metadata,
        "version": TEACHER_DATASET_VERSION,
        "action_size": ACTION_SIZE,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            **dataset.columns,
            observation_shape=np.asarray(dataset.observation_shape, dtype=np.int16),
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=True)),
        )
    temporary.replace(output)
    return output


def load_teacher_dataset(path: str | Path) -> TeacherDataset:
    with np.load(path, allow_pickle=False) as document:
        columns = {
            name: np.asarray(document[name], dtype=dtype)
            for name, dtype in COLUMN_DTYPES.items()
        }
        shape = tuple(int(value) for value in document["observation_shape"])
        metadata = json.loads(str(document["metadata"].item()))
    if not isinstance(metadata, dict) or (
        metadata.get("version") != TEACHER_DATASET_VERSION
        or metadata.get("action_size") != ACTION_SIZE
    ):
        raise ValueError(f"unsupported teacher dataset format: {path}")
    dataset = TeacherDataset(columns, shape, metadata)
    validate_teacher_dataset(dataset)
    return dataset


def validate_teacher_dataset(dataset: TeacherDataset) -> None:
    shape = dataset.observation_shape
    if len(shape) != 3 or shape[0] < 18 or shape[0] % 18 or shape[1:] != (8, 8):
        raise ValueError("teacher observations must have shape (18 * frames, 8, 8)")
    if set(dataset.columns) != set(COLUMN_DTYPES):
        raise ValueError("missing or unexpected teacher dataset columns")
    count = len(dataset)
    for name, values in dataset.columns.items():
        expected_dtype = np.dtype(COLUMN_DTYPES[name])
        if (expected_dtype.kind == "U" and values.dtype.kind != "U") or (
            expected_dtype.kind != "U" and values.dtype != expected_dtype
        ):
            raise ValueError(f"invalid teacher column dtype: {name}")
        if values.shape != _column_shape(name, count, shape):
            raise ValueError(f"invalid teacher column shape: {name}")
    masks = dataset.columns["packed_action_masks"]
    for name in ("teacher_actions", "played_actions"):
        actions = dataset.columns[name]
        if np.any((actions < 0) | (actions >= ACTION_SIZE)):
            raise ValueError(f"invalid action id: {name}")
        legal = (masks[np.arange(count), actions // 8] >> (actions % 8)) & 1
        if not np.all(legal):
            raise ValueError(f"illegal action in teacher dataset: {name}")
    for prefix in ("teacher", "played"):
        cp = dataset.columns[f"{prefix}_cp"]
        mate = dataset.columns[f"{prefix}_mate"]
        if np.any(np.isinf(cp) | np.isinf(mate)) or not np.all(
            np.isfinite(cp) ^ np.isfinite(mate)
        ):
            raise ValueError("each engine score must contain either cp or mate")
    if np.any(dataset.columns["game_ids"] < 1) or np.any(dataset.columns["plies"] < 0):
        raise ValueError("invalid game id or ply")
