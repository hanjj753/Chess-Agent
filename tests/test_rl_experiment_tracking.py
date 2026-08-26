import csv
import json
from pathlib import Path

from chess_agent.rl.experiment_tracking import ExperimentLogger


def test_experiment_logger_records_config_metrics_games_and_summary(
    tmp_path: Path,
) -> None:
    logger = ExperimentLogger.create(
        tmp_path,
        experiment_name="full chess / PPO",
        config={"learning_rate": 0.0003, "checkpoint": Path("tmp/model.pt")},
        run_id="test_run",
    )

    logger.log_metrics(
        step=10,
        phase="train",
        metrics={"policy_loss": 0.5, "win_rate": 0.25},
    )
    logger.log_game(
        step=10,
        phase="evaluation",
        episode=1,
        result="1-0",
        reward=1.0,
        plies=63,
        agent_color="white",
        opponent="random",
        termination="checkmate",
    )
    logger.log_checkpoint(step=10, path="tmp/model.pt", is_best=True)
    logger.save_summary({"episodes": 1, "win_rate": 0.25})

    assert logger.run_dir.name == "test_run_full_chess_PPO"
    config = json.loads((logger.run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["config"]["checkpoint"] == str(Path("tmp/model.pt"))

    with logger.metrics_path.open(encoding="utf-8", newline="") as source:
        metric_rows = list(csv.DictReader(source))
    assert [row["metric"] for row in metric_rows] == ["policy_loss", "win_rate"]
    assert {row["step"] for row in metric_rows} == {"10"}

    with logger.games_path.open(encoding="utf-8", newline="") as source:
        game_rows = list(csv.DictReader(source))
    assert game_rows[0]["result"] == "1-0"
    assert game_rows[0]["plies"] == "63"
    assert game_rows[0]["phase"] == "evaluation"

    events = [
        json.loads(line)
        for line in logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "experiment_started",
        "metrics",
        "checkpoint",
        "experiment_completed",
    ]
    assert logger.summary_path.exists()
