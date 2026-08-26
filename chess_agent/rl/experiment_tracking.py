import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping


METRIC_FIELDS = ("timestamp", "step", "phase", "metric", "value")
GAME_FIELDS = (
    "timestamp",
    "step",
    "phase",
    "episode",
    "result",
    "reward",
    "plies",
    "agent_color",
    "opponent",
    "termination",
    "checkpoint",
)


class ExperimentLogger:
    """Append-only experiment records suitable for analysis and presentation plots."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.games_path = self.run_dir / "games.csv"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        ensure_csv_header(self.metrics_path, METRIC_FIELDS)
        ensure_csv_header(self.games_path, GAME_FIELDS)

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        *,
        experiment_name: str,
        config: Any,
        run_id: str | None = None,
    ) -> "ExperimentLogger":
        safe_name = sanitize_name(experiment_name)
        resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger = cls(Path(base_dir) / f"{resolved_run_id}_{safe_name}")
        logger.write_json(
            logger.run_dir / "config.json",
            {
                "experiment_name": experiment_name,
                "run_id": resolved_run_id,
                "created_at": utc_timestamp(),
                "config": to_jsonable(config),
            },
        )
        logger.log_event("experiment_started")
        return logger

    def log_metrics(
        self,
        *,
        step: int,
        phase: str,
        metrics: Mapping[str, int | float],
    ) -> None:
        timestamp = utc_timestamp()
        rows = []
        normalized_metrics: dict[str, int | float] = {}
        for metric, raw_value in metrics.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(f"metric must be numeric: {metric}")
            value = int(raw_value) if isinstance(raw_value, int) else float(raw_value)
            normalized_metrics[str(metric)] = value
            rows.append(
                {
                    "timestamp": timestamp,
                    "step": int(step),
                    "phase": phase,
                    "metric": str(metric),
                    "value": value,
                }
            )
        append_csv_rows(self.metrics_path, METRIC_FIELDS, rows)
        self.log_event(
            "metrics",
            step=int(step),
            phase=phase,
            metrics=normalized_metrics,
        )

    def log_game(
        self,
        *,
        step: int,
        phase: str,
        episode: int,
        result: str,
        reward: float,
        plies: int,
        agent_color: str,
        opponent: str,
        termination: str,
        checkpoint: str | Path | None = None,
    ) -> None:
        append_csv_rows(
            self.games_path,
            GAME_FIELDS,
            [
                {
                    "timestamp": utc_timestamp(),
                    "step": int(step),
                    "phase": phase,
                    "episode": int(episode),
                    "result": result,
                    "reward": float(reward),
                    "plies": int(plies),
                    "agent_color": agent_color,
                    "opponent": opponent,
                    "termination": termination,
                    "checkpoint": "" if checkpoint is None else str(checkpoint),
                }
            ],
        )

    def log_checkpoint(
        self,
        *,
        step: int,
        path: str | Path,
        is_best: bool = False,
        metrics: Mapping[str, int | float] | None = None,
    ) -> None:
        self.log_event(
            "checkpoint",
            step=int(step),
            path=str(path),
            is_best=bool(is_best),
            metrics=dict(metrics or {}),
        )

    def save_summary(self, summary: Any) -> Path:
        self.write_json(
            self.summary_path,
            {
                "completed_at": utc_timestamp(),
                "summary": to_jsonable(summary),
            },
        )
        self.log_event("experiment_completed", summary=to_jsonable(summary))
        return self.summary_path

    def log_event(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": utc_timestamp(),
            "event": event,
            **to_jsonable(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


def ensure_csv_header(path: Path, fieldnames: tuple[str, ...]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        csv.DictWriter(output, fieldnames=fieldnames).writeheader()


def append_csv_rows(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writerows(rows)


def sanitize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "experiment"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
