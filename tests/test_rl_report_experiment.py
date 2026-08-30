from pathlib import Path

from chess_agent.rl.experiment_tracking import ExperimentLogger
from chess_agent.rl.report_experiment import (
    generate_experiment_report,
    generate_experiment_reports,
)


def test_generate_experiment_report_from_one_experiment(
    tmp_path: Path,
) -> None:
    logger = ExperimentLogger.create(
        tmp_path,
        experiment_name="ppo report",
        run_id="test_run",
        config={
            "total_timesteps": 512,
            "opponent": "alpha-random",
            "alpha_move_probability": 0.25,
            "opponent_depth": 1,
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
    logger.log_metrics(
        step=256,
        phase="rollout",
        metrics={
            "transitions": 256,
            "completed_games": 4,
            "decisive_games": 2,
            "reward_signal_rate": 0.0078125,
            "return_std": 0.4,
            "value_prediction_std": 0.1,
            "advantage_std": 0.5,
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
        opponent="alpha-random",
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
        opponent="alpha-random",
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
        opponent="alpha-random",
        termination="checkmate",
    )
    logger.log_checkpoint(
        step=256,
        path="tmp/full_chess_ppo_best.zip",
        is_best=True,
        metrics={"score_rate": 0.0},
    )
    logger.save_summary({"completed_timesteps": 256})

    result = generate_experiment_report(logger.run_dir)

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
    assert "완결 대국 수" in report
    assert "rollout당 완결 대국이 평균 8판보다 적어" in report
    assert "reward가 0이 아닌 transition이 평균 1%보다 적어" in report
    assert "alpha_move_probability=25.0%" in report

    assert generate_experiment_reports(logger.run_dir) == ()
    result.game_outcomes_path.unlink()
    regenerated = generate_experiment_reports(logger.run_dir)
    assert len(regenerated) == 1
    assert regenerated[0].game_outcomes_path is not None
    assert regenerated[0].game_outcomes_path.is_file()


def test_generate_experiment_reports_for_every_child_experiment(
    tmp_path: Path,
) -> None:
    experiment_names = ("first", "second")
    for experiment_name in experiment_names:
        logger = ExperimentLogger.create(
            tmp_path,
            experiment_name=experiment_name,
            run_id=f"run_{experiment_name}",
            config={"total_timesteps": 1, "opponent": "random"},
        )
        logger.log_metrics(
            step=1,
            phase="train_update",
            metrics={"value_loss": 0.1},
        )
        logger.log_game(
            step=1,
            phase="train",
            episode=1,
            result="1-0",
            reward=1.0,
            plies=1,
            agent_color="white",
            opponent="random",
            termination="checkmate",
        )

    output_dir = tmp_path / "reports"
    results = generate_experiment_reports(
        tmp_path,
        output_dir=output_dir,
        create_plots=False,
    )

    assert [result.experiment_dir.name for result in results] == [
        "run_first_first",
        "run_second_second",
    ]
    assert all(result.summary_path.is_file() for result in results)
    assert [result.output_dir for result in results] == [
        output_dir / "run_first_first",
        output_dir / "run_second_second",
    ]


def test_generate_experiment_reports_skips_complete_reports_unless_forced(
    tmp_path: Path,
) -> None:
    logger = ExperimentLogger.create(
        tmp_path,
        experiment_name="skip existing",
        run_id="test_run",
        config={"total_timesteps": 1, "opponent": "random"},
    )
    logger.log_metrics(
        step=1,
        phase="train_update",
        metrics={"value_loss": 0.1},
    )

    first_results = generate_experiment_reports(tmp_path, create_plots=False)
    assert len(first_results) == 1
    summary_path = first_results[0].summary_path
    summary_path.write_text("keep this report", encoding="utf-8")

    skipped_results = generate_experiment_reports(tmp_path, create_plots=False)
    assert skipped_results == ()
    assert summary_path.read_text(encoding="utf-8") == "keep this report"

    forced_results = generate_experiment_reports(
        tmp_path,
        create_plots=False,
        force=True,
    )
    assert len(forced_results) == 1
    assert summary_path.read_text(encoding="utf-8") != "keep this report"
