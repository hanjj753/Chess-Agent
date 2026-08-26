from pathlib import Path

from chess_agent.rl.experiment_tracking import ExperimentLogger
from chess_agent.rl.report_experiment import generate_experiment_report


def test_generate_experiment_report_from_latest_experiment(
    tmp_path: Path,
) -> None:
    logger = ExperimentLogger.create(
        tmp_path,
        experiment_name="ppo report",
        run_id="test_run",
        config={
            "total_timesteps": 512,
            "opponent": "random",
            "learning_rate": 0.0001,
            "n_envs": 1,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 2,
            "target_kl": 0.03,
            "max_plies": 100,
        },
    )
    logger.log_metrics(
        step=256,
        phase="train_update",
        metrics={
            "policy_loss": 0.2,
            "value_loss": 0.1,
            "entropy": 1.1,
            "approx_kl": 0.3,
            "clip_fraction": 0.5,
            "explained_variance": -0.2,
        },
    )
    logger.log_game(
        step=20,
        phase="train",
        episode=1,
        result="1-0",
        reward=1.0,
        plies=39,
        agent_color="white",
        opponent="random",
        termination="checkmate",
    )
    logger.log_game(
        step=50,
        phase="train",
        episode=2,
        result="1/2-1/2",
        reward=0.0,
        plies=100,
        agent_color="black",
        opponent="random",
        termination="max_plies",
    )
    logger.log_game(
        step=256,
        phase="evaluation",
        episode=1,
        result="0-1",
        reward=-1.0,
        plies=44,
        agent_color="white",
        opponent="random",
        termination="checkmate",
    )
    logger.log_checkpoint(
        step=256,
        path="tmp/full_chess_ppo_best.zip",
        is_best=True,
        metrics={"score_rate": 0.0},
    )
    logger.save_summary({"completed_timesteps": 256})

    result = generate_experiment_report(tmp_path)

    assert result.experiment_dir == logger.run_dir
    assert result.summary_path.is_file()
    assert result.learning_curves_path is not None
    assert result.learning_curves_path.stat().st_size > 1_000
    assert result.game_outcomes_path is not None
    assert result.game_outcomes_path.stat().st_size > 1_000

    report = result.summary_path.read_text(encoding="utf-8")
    assert "episode 1개는 현재 환경에서 체스 대국 1판입니다." in report
    assert "W/D/L:           1/1/0" in report
    assert "평균 approx_kl이 target_kl의 1.5배인 0.0450보다 큽니다." in report
    assert "Critic이 아직 유용한 value 예측을 하지 못합니다." in report
