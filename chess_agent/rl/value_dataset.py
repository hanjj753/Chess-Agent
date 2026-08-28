from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


VALUE_DATASET_VERSION = 1


@dataclass(frozen=True)
class PackedValueDataset:
    packed_observations: np.ndarray
    targets: np.ndarray
    outcomes: np.ndarray
    game_ids: np.ndarray
    observation_shape: tuple[int, int, int]
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    @property
    def games(self) -> int:
        return int(np.unique(self.game_ids).size)

    def unpack(self, indices: np.ndarray) -> np.ndarray:
        packed = self.packed_observations[indices]
        feature_count = int(np.prod(self.observation_shape))
        observations = np.unpackbits(
            packed,
            axis=1,
            count=feature_count,
            bitorder="little",
        )
        return observations.reshape((-1, *self.observation_shape)).astype(
            np.float32,
            copy=False,
        )


def pack_observations(observations: np.ndarray) -> np.ndarray:
    array = np.asarray(observations)
    if array.ndim != 4 or array.shape[-2:] != (8, 8):
        raise ValueError("observations must have shape (samples, channels, 8, 8)")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("observations must contain only 0 and 1")
    flattened = array.astype(np.uint8, copy=False).reshape(array.shape[0], -1)
    return np.packbits(flattened, axis=1, bitorder="little")


def save_value_dataset(
    path: str | Path,
    *,
    packed_observations: np.ndarray,
    targets: np.ndarray,
    outcomes: np.ndarray,
    game_ids: np.ndarray,
    observation_shape: tuple[int, int, int],
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_metadata = dict(metadata or {})
    normalized_metadata["version"] = VALUE_DATASET_VERSION
    np.savez_compressed(
        output_path,
        packed_observations=np.asarray(packed_observations, dtype=np.uint8),
        targets=np.asarray(targets, dtype=np.float32),
        outcomes=np.asarray(outcomes, dtype=np.int8),
        game_ids=np.asarray(game_ids, dtype=np.int32),
        observation_shape=np.asarray(observation_shape, dtype=np.int16),
        metadata=np.asarray(json.dumps(normalized_metadata, ensure_ascii=True)),
    )
    return output_path


def load_value_dataset(path: str | Path) -> PackedValueDataset:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as document:
        packed_observations = np.asarray(
            document["packed_observations"],
            dtype=np.uint8,
        )
        targets = np.asarray(document["targets"], dtype=np.float32)
        outcomes = np.asarray(document["outcomes"], dtype=np.int8)
        game_ids = np.asarray(document["game_ids"], dtype=np.int32)
        shape_values = np.asarray(document["observation_shape"], dtype=np.int64)
        raw_metadata = str(document["metadata"].item())

    if shape_values.shape != (3,):
        raise ValueError(f"invalid observation shape in value dataset: {input_path}")
    observation_shape = tuple(int(value) for value in shape_values)
    metadata = json.loads(raw_metadata)
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid metadata in value dataset: {input_path}")
    if int(metadata.get("version", -1)) != VALUE_DATASET_VERSION:
        raise ValueError(f"unsupported value dataset version: {input_path}")

    sample_count = int(targets.shape[0])
    if targets.ndim != 1:
        raise ValueError(f"value targets must be one-dimensional: {input_path}")
    if outcomes.shape != (sample_count,) or game_ids.shape != (sample_count,):
        raise ValueError(f"value dataset columns have different lengths: {input_path}")
    if packed_observations.ndim != 2 or packed_observations.shape[0] != sample_count:
        raise ValueError(f"invalid packed observations in: {input_path}")
    expected_bytes = (int(np.prod(observation_shape)) + 7) // 8
    if packed_observations.shape[1] != expected_bytes:
        raise ValueError(f"packed observation width does not match shape: {input_path}")
    if not np.all(np.isin(outcomes, (-1, 0, 1))):
        raise ValueError(f"outcomes must be -1, 0, or 1: {input_path}")

    return PackedValueDataset(
        packed_observations=packed_observations,
        targets=targets,
        outcomes=outcomes,
        game_ids=game_ids,
        observation_shape=observation_shape,
        metadata=metadata,
    )
