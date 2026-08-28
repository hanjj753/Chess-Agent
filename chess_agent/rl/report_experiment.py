import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "chess_agent_matplotlib"),
)
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MetricRow:
    timestamp: str
    step: int
    phase: str
    metric: str
    value: float


@dataclass(frozen=True)
class GameRow:
    timestamp: str
    step: int
    phase: str
    episode: int
    result: str
    reward: float
    plies: int
    agent_color: str
    opponent: str
    termination: str
    checkpoint: str


@dataclass(frozen=True)
class ExperimentData:
    directory: Path
    config_document: dict[str, Any]
    metrics: tuple[MetricRow, ...]
    games: tuple[GameRow, ...]
    events: tuple[dict[str, Any], ...]
    summary_document: dict[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        config = self.config_document.get("config", {})
        return config if isinstance(config, dict) else {}


@dataclass(frozen=True)
class GameStats:
    games: int
    wins: int
    draws: int
    losses: int
    score_rate: float
    average_plies: float
    terminations: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GeneratedExperimentReport:
    experiment_dir: Path
    output_dir: Path
    summary_path: Path
    learning_curves_path: Path | None
    game_outcomes_path: Path | None


def generate_experiment_report(
    experiment_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    create_plots: bool = True,
) -> GeneratedExperimentReport:
    experiment_dir = resolve_experiment_dir(experiment_path)
    data = load_experiment(experiment_dir)
    value_pretraining = is_value_pretraining_experiment(data)
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else experiment_dir / "report"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    report_text = (
        build_value_pretraining_text_report(data)
        if value_pretraining
        else build_text_report(data)
    )
    summary_path = resolved_output_dir / "summary.txt"
    summary_path.write_text(report_text, encoding="utf-8")

    learning_curves_path = None
    game_outcomes_path = None
    if create_plots:
        learning_curves_path = resolved_output_dir / "learning_curves.png"
        if value_pretraining:
            plot_value_pretraining_curves(data, learning_curves_path)
        else:
            game_outcomes_path = resolved_output_dir / "game_outcomes.png"
            plot_learning_curves(data, learning_curves_path)
            plot_game_outcomes(data, game_outcomes_path)

    return GeneratedExperimentReport(
        experiment_dir=experiment_dir,
        output_dir=resolved_output_dir,
        summary_path=summary_path,
        learning_curves_path=learning_curves_path,
        game_outcomes_path=game_outcomes_path,
    )


def generate_experiment_reports(
    experiment_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    create_plots: bool = True,
) -> tuple[GeneratedExperimentReport, ...]:
    experiment_dirs = resolve_experiment_dirs(experiment_path)
    shared_output_dir = Path(output_dir) if output_dir is not None else None
    use_named_output_dirs = shared_output_dir is not None and len(experiment_dirs) > 1

    reports = []
    for experiment_dir in experiment_dirs:
        report_output_dir = shared_output_dir
        if use_named_output_dirs:
            report_output_dir = shared_output_dir / experiment_dir.name
        reports.append(
            generate_experiment_report(
                experiment_dir,
                output_dir=report_output_dir,
                create_plots=create_plots,
            )
        )
    return tuple(reports)


def resolve_experiment_dirs(path: str | Path) -> tuple[Path, ...]:
    candidate = Path(path)
    if (candidate / "config.json").is_file():
        return (candidate,)
    if not candidate.is_dir():
        raise ValueError(f"experiment path is not a directory: {candidate}")

    experiments = tuple(
        sorted(
            (
                child
                for child in candidate.iterdir()
                if child.is_dir() and (child / "config.json").is_file()
            ),
            key=lambda child: child.name,
        )
    )
    if not experiments:
        raise ValueError(f"no experiment directories found under: {candidate}")
    return experiments


def resolve_experiment_dir(path: str | Path) -> Path:
    experiment_dirs = resolve_experiment_dirs(path)
    if len(experiment_dirs) != 1:
        raise ValueError(
            "expected one experiment directory; "
            "use generate_experiment_reports() for a parent directory"
        )
    return experiment_dirs[0]


def load_experiment(directory: str | Path) -> ExperimentData:
    experiment_dir = Path(directory)
    config_path = experiment_dir / "config.json"
    metrics_path = experiment_dir / "metrics.csv"
    games_path = experiment_dir / "games.csv"
    if not config_path.is_file():
        raise ValueError(f"missing config.json in: {experiment_dir}")
    if not metrics_path.is_file():
        raise ValueError(f"missing metrics.csv in: {experiment_dir}")
    if not games_path.is_file():
        raise ValueError(f"missing games.csv in: {experiment_dir}")

    return ExperimentData(
        directory=experiment_dir,
        config_document=read_json(config_path),
        metrics=read_metrics(metrics_path),
        games=read_games(games_path),
        events=read_json_lines(experiment_dir / "events.jsonl"),
        summary_document=read_optional_json(experiment_dir / "summary.json"),
    )


def build_text_report(data: ExperimentData) -> str:
    config = data.config
    experiment_name = str(
        data.config_document.get("experiment_name", data.directory.name)
    )
    completed_step = completed_timesteps(data)
    train_games = tuple(game for game in data.games if game.phase == "train")
    train_stats = summarize_games(train_games)
    evaluation_groups = grouped_evaluation_games(data.games)
    diagnostic_names = (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    )
    diagnostics = {
        name: metric_values(data.metrics, phase="train_update", metric=name)
        for name in diagnostic_names
    }
    rollout_names = (
        "transitions",
        "completed_games",
        "decisive_games",
        "reward_signal_rate",
        "return_std",
        "value_prediction_std",
        "advantage_std",
    )
    rollout_diagnostics = {
        name: metric_values(data.metrics, phase="rollout", metric=name)
        for name in rollout_names
    }

    lines = [
        "Full-Chess PPO 실험 자동 보고서",
        "=" * 34,
        f"실험 이름:       {experiment_name}",
        f"실험 폴더:       {data.directory}",
        f"완료 timestep:   {completed_step:,}",
        f"목표 timestep:   {format_int(config.get('total_timesteps'))}",
        f"학습 상대:       {config.get('opponent', 'unknown')}",
        f"학습률:          {config.get('learning_rate', 'unknown')}",
        f"rollout:         n_envs={config.get('n_envs', '?')} x n_steps={config.get('n_steps', '?')}",
        f"PPO update:      batch={config.get('batch_size', '?')}, epochs={config.get('n_epochs', '?')}",
        f"PPO limits:      clip={config.get('clip_range', '?')}, target_kl={config.get('target_kl', '?')}",
        f"게임 제한:       max_plies={config.get('max_plies', '?')}",
        f"소요 시간:       {experiment_duration(data)}",
        "",
        "단위 설명",
        "---------",
        "episode 1개는 현재 환경에서 체스 대국 1판입니다.",
        "timestep 1개는 agent가 수 1개를 선택한 순간이며, 환경이 상대 응수까지 진행합니다.",
        "rollout은 PPO update 전에 모으는 여러 timestep 묶음이고, 여러 episode가 섞일 수 있습니다.",
        "evaluation 대국은 학습 데이터를 만들지 않고 현재 실력만 측정합니다.",
        "",
        "학습 대국 요약",
        "--------------",
        f"대국 수:         {train_stats.games}",
        f"W/D/L:           {train_stats.wins}/{train_stats.draws}/{train_stats.losses}",
        f"점수율:          {train_stats.score_rate:.1%}",
        f"평균 game plies: {train_stats.average_plies:.1f}",
    ]
    for termination, count in train_stats.terminations:
        rate = count / train_stats.games if train_stats.games else 0.0
        lines.append(f"종료 {termination:20s} {count:5d} ({rate:5.1%})")

    lines.extend(
        [
            "",
            "평가 스냅샷",
            "-------------",
            "구분                 step  games       W/D/L   score  avg plies",
        ]
    )
    if evaluation_groups:
        for (phase, step), games in evaluation_groups:
            stats = summarize_games(games)
            lines.append(
                f"{phase:18s} {step:8,d} {stats.games:6d} "
                f"{stats.wins:3d}/{stats.draws:3d}/{stats.losses:3d} "
                f"{stats.score_rate:7.1%} {stats.average_plies:10.1f}"
            )
    else:
        lines.append("평가 대국 기록이 없습니다.")

    best_checkpoint = find_best_checkpoint(data.events)
    lines.extend(["", "Best checkpoint", "---------------"])
    if best_checkpoint is None:
        lines.append("기록된 best checkpoint가 없습니다.")
    else:
        lines.append(f"step:             {int(best_checkpoint.get('step', 0)):,}")
        lines.append(f"path:             {best_checkpoint.get('path', '')}")
        best_metrics = best_checkpoint.get("metrics", {})
        if isinstance(best_metrics, dict) and "score_rate" in best_metrics:
            lines.append(f"evaluation score: {float(best_metrics['score_rate']):.1%}")

    lines.extend(["", "PPO 학습 진단", "-------------"])
    for name in diagnostic_names:
        values = diagnostics[name]
        if not values:
            continue
        lines.append(
            f"{name:20s} avg={mean(values):9.4f} "
            f"min={min(values):9.4f} max={max(values):9.4f} "
            f"last={values[-1]:9.4f}"
        )

    lines.extend(["", "Rollout과 Critic 입력", "-------------------"])
    rollout_labels = {
        "transitions": "transition 수",
        "completed_games": "완결 대국 수",
        "decisive_games": "승패 대국 수",
        "reward_signal_rate": "reward 신호 비율",
        "return_std": "return 표준편차",
        "value_prediction_std": "value 예측 표준편차",
        "advantage_std": "advantage 표준편차",
    }
    has_rollout_metrics = False
    for name in rollout_names:
        values = rollout_diagnostics[name]
        if not values:
            continue
        has_rollout_metrics = True
        if name == "reward_signal_rate":
            lines.append(
                f"{rollout_labels[name]:20s} avg={mean(values):8.2%} "
                f"min={min(values):8.2%} max={max(values):8.2%} "
                f"last={values[-1]:8.2%}"
            )
        else:
            lines.append(
                f"{rollout_labels[name]:20s} avg={mean(values):9.3f} "
                f"min={min(values):9.3f} max={max(values):9.3f} "
                f"last={values[-1]:9.3f}"
            )
    if not has_rollout_metrics:
        lines.append("이 실험에는 rollout 진단 기록이 없습니다.")

    warnings = build_warnings(
        data,
        evaluation_groups,
        diagnostics,
        rollout_diagnostics,
    )
    lines.extend(
        [
            "",
            "해석과 주의사항",
            "----------------",
            "아래 임계값은 빠른 점검용 경험적 경고이며 절대적인 합격 기준은 아닙니다.",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- 자동 진단에서 뚜렷한 경고가 발견되지 않았습니다.")

    lines.extend(
        [
            "",
            "빠르게 볼 항목",
            "--------------",
            "1. evaluation score가 step에 따라 실제로 상승하는가?",
            "2. max_plies 무승부가 줄고 checkmate 승리가 늘어나는가?",
            "3. approx_kl과 clip_fraction이 지나치게 크지 않은가?",
            "4. explained_variance가 0보다 높아지며 Critic이 학습되는가?",
            "5. rollout마다 완결/승패 대국과 reward 신호가 충분히 들어오는가?",
            "6. illegal action이 항상 0인가?",
            "",
        ]
    )
    return "\n".join(lines)


def is_value_pretraining_experiment(data: ExperimentData) -> bool:
    return (
        "train_data_path" in data.config
        and "validation_data_path" in data.config
        and any(row.phase == "value_pretrain" for row in data.metrics)
    )


def build_value_pretraining_text_report(data: ExperimentData) -> str:
    config = data.config
    experiment_name = str(
        data.config_document.get("experiment_name", data.directory.name)
    )
    metric_names = (
        "train_loss",
        "validation_loss",
        "validation_mae",
        "validation_rmse",
        "validation_explained_variance",
        "validation_target_std",
        "validation_prediction_std",
    )
    diagnostics = {
        name: metric_values(data.metrics, phase="value_pretrain", metric=name)
        for name in metric_names
    }
    completed_epoch = max(
        (row.step for row in data.metrics if row.phase == "value_pretrain"),
        default=0,
    )
    best_checkpoint = find_best_checkpoint(data.events)
    lines = [
        "Value-head supervised pretraining 보고서",
        "=" * 39,
        f"실험 이름:       {experiment_name}",
        f"실험 폴더:       {data.directory}",
        f"source model:    {config.get('model_path', 'unknown')}",
        f"train data:      {config.get('train_data_path', 'unknown')}",
        f"validation data: {config.get('validation_data_path', 'unknown')}",
        f"완료 epoch:      {completed_epoch}",
        f"목표 epoch:      {format_int(config.get('epochs'))}",
        f"batch size:      {format_int(config.get('batch_size'))}",
        f"learning rate:   {config.get('learning_rate', 'unknown')}",
        f"대국 균형:       {config.get('balance_games', 'unknown')}",
        f"승무패 균형:     {config.get('balance_outcomes', 'unknown')}",
        f"소요 시간:       {experiment_duration(data)}",
        "",
        "학습 지표",
        "---------",
    ]
    for name in metric_names:
        values = diagnostics[name]
        if not values:
            continue
        lines.append(
            f"{name:32s} initial={values[0]:9.5f} "
            f"min={min(values):9.5f} max={max(values):9.5f} "
            f"last={values[-1]:9.5f}"
        )
    lines.extend(["", "Best checkpoint", "---------------"])
    if best_checkpoint is None:
        lines.append("기록된 best checkpoint가 없습니다.")
    else:
        lines.append(f"epoch:            {int(best_checkpoint.get('step', 0))}")
        lines.append(f"path:             {best_checkpoint.get('path', '')}")
        best_metrics = best_checkpoint.get("metrics", {})
        if isinstance(best_metrics, dict) and "validation_loss" in best_metrics:
            lines.append(
                f"validation loss:  {float(best_metrics['validation_loss']):.5f}"
            )

    lines.extend(["", "해석과 주의사항", "----------------"])
    explained_variance = diagnostics["validation_explained_variance"]
    prediction_std = diagnostics["validation_prediction_std"]
    target_std = diagnostics["validation_target_std"]
    if explained_variance and explained_variance[-1] <= 0:
        lines.append("- validation explained variance가 0 이하라 value가 아직 target을 설명하지 못합니다.")
    if prediction_std and target_std and prediction_std[-1] < 0.25 * target_std[-1]:
        lines.append("- value prediction 표준편차가 target의 25%보다 작아 예측이 지나치게 평평합니다.")
    if (
        explained_variance
        and explained_variance[-1] > 0
        and prediction_std
        and target_std
        and prediction_std[-1] >= 0.25 * target_std[-1]
    ):
        lines.append("- value 예측이 target 차이를 양의 explained variance로 설명하기 시작했습니다.")
    lines.extend(
        [
            "",
            "빠르게 볼 항목",
            "--------------",
            "1. validation loss와 MAE가 감소하는가?",
            "2. explained variance가 0보다 높아지는가?",
            "3. prediction std가 0에 머물지 않고 target std에 가까워지는가?",
            "4. best epoch 이후 validation 성능이 악화되어 과적합하지 않는가?",
            "",
        ]
    )
    return "\n".join(lines)


def build_warnings(
    data: ExperimentData,
    evaluation_groups: list[tuple[tuple[str, int], tuple[GameRow, ...]]],
    diagnostics: dict[str, list[float]],
    rollout_diagnostics: dict[str, list[float]],
) -> list[str]:
    warnings: list[str] = []
    periodic_steps = [
        step for (phase, step), _ in evaluation_groups if phase == "evaluation"
    ]
    is_resumed = bool(data.config.get("resume_from"))
    if not is_resumed and 0 not in periodic_steps:
        warnings.append(
            "step=0 학습 전 평가가 없어 PPO가 pretrained policy를 개선했는지 직접 비교할 수 없습니다."
        )
    if is_resumed and not periodic_steps:
        warnings.append(
            "재개 시점의 학습 전 평가가 없어 resume checkpoint 대비 개선 여부를 직접 비교할 수 없습니다."
        )

    evaluation_sizes = [len(games) for _, games in evaluation_groups]
    if evaluation_sizes and min(evaluation_sizes) < 200:
        warnings.append(
            "평가 대국이 200판보다 적은 시점이 있어 checkpoint 선택이 표본 변동에 흔들릴 수 있습니다."
        )

    final_groups = [
        games for (phase, _), games in evaluation_groups if phase == "final_evaluation"
    ]
    if final_groups:
        final_stats = summarize_games(final_groups[-1])
        max_plies_count = dict(final_stats.terminations).get("max_plies", 0)
        if final_stats.games and max_plies_count / final_stats.games >= 0.3:
            warnings.append(
                "최종 평가의 max_plies 종료가 30% 이상입니다. 이 무승부는 실력 향상으로 해석하면 안 됩니다."
            )

    approx_kl = diagnostics.get("approx_kl", [])
    configured_target_kl = data.config.get("target_kl")
    target_kl = (
        float(configured_target_kl)
        if isinstance(configured_target_kl, (int, float))
        else None
    )
    kl_warning_threshold = 1.5 * target_kl if target_kl is not None else 0.1
    if approx_kl and mean(approx_kl) > kl_warning_threshold:
        if target_kl is None:
            warnings.append(
                "평균 approx_kl이 0.1보다 커 policy가 update마다 크게 변하고 있습니다."
            )
        else:
            warnings.append(
                f"평균 approx_kl이 target_kl의 1.5배인 {kl_warning_threshold:.4f}보다 큽니다."
            )
    clip_fraction = diagnostics.get("clip_fraction", [])
    if clip_fraction and mean(clip_fraction) > 0.4:
        warnings.append(
            "평균 clip_fraction이 40%보다 커 대부분의 PPO sample이 clipping되고 있습니다."
        )
    explained_variance = diagnostics.get("explained_variance", [])
    if explained_variance and mean(explained_variance) < 0:
        warnings.append(
            "평균 explained_variance가 음수여서 Critic이 아직 유용한 value 예측을 하지 못합니다."
        )

    completed_games = rollout_diagnostics.get("completed_games", [])
    if completed_games and mean(completed_games) < 8:
        warnings.append(
            "rollout당 완결 대국이 평균 8판보다 적어 Critic target의 표본 변동이 클 수 있습니다."
        )
    reward_signal_rate = rollout_diagnostics.get("reward_signal_rate", [])
    if reward_signal_rate and mean(reward_signal_rate) < 0.01:
        warnings.append(
            "reward가 0이 아닌 transition이 평균 1%보다 적어 학습 신호가 매우 희소합니다."
        )

    illegal_actions = sum(
        1
        for game in data.games
        if game.termination == "illegal_action" or game.result == "illegal_action"
    )
    if illegal_actions:
        warnings.append(f"불법 수로 종료된 대국이 {illegal_actions}판 있습니다.")
    else:
        warnings.append("불법 수 종료가 0판이므로 action masking은 정상 동작했습니다.")
    return warnings


def plot_value_pretraining_curves(
    data: ExperimentData,
    output_path: str | Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle(f"Value Pretraining - {data.directory.name}", fontsize=14)

    loss_axis = axes[0, 0]
    plot_metric(
        loss_axis,
        data.metrics,
        "train_loss",
        label="Train loss",
        phase="value_pretrain",
    )
    plot_metric(
        loss_axis,
        data.metrics,
        "validation_loss",
        label="Validation loss",
        phase="value_pretrain",
    )
    loss_axis.set_title("Huber loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    error_axis = axes[0, 1]
    plot_metric(
        error_axis,
        data.metrics,
        "validation_mae",
        label="Validation MAE",
        phase="value_pretrain",
    )
    plot_metric(
        error_axis,
        data.metrics,
        "validation_rmse",
        label="Validation RMSE",
        phase="value_pretrain",
    )
    error_axis.set_title("Validation error")
    error_axis.set_xlabel("Epoch")
    error_axis.grid(alpha=0.25)
    error_axis.legend()

    explained_axis = axes[1, 0]
    plot_metric(
        explained_axis,
        data.metrics,
        "validation_explained_variance",
        label="Explained variance",
        phase="value_pretrain",
    )
    explained_axis.axhline(0, color="#777777", linewidth=1, alpha=0.6)
    explained_axis.set_title("Value explained variance")
    explained_axis.set_xlabel("Epoch")
    explained_axis.grid(alpha=0.25)
    explained_axis.legend()

    spread_axis = axes[1, 1]
    plot_metric(
        spread_axis,
        data.metrics,
        "validation_target_std",
        label="Target std",
        phase="value_pretrain",
    )
    plot_metric(
        spread_axis,
        data.metrics,
        "validation_prediction_std",
        label="Prediction std",
        phase="value_pretrain",
    )
    spread_axis.set_title("Value target and prediction spread")
    spread_axis.set_xlabel("Epoch")
    spread_axis.grid(alpha=0.25)
    spread_axis.legend()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_learning_curves(data: ExperimentData, output_path: str | Path) -> Path:
    has_rollout_metrics = any(row.phase == "rollout" for row in data.metrics)
    plot_rows = 3 if has_rollout_metrics else 2
    figure, axes = plt.subplots(
        plot_rows,
        2,
        figsize=(12, 4 * plot_rows),
        constrained_layout=True,
    )
    figure.suptitle(f"PPO Learning Curves - {data.directory.name}", fontsize=14)

    periodic = [
        (step, summarize_games(games))
        for (phase, step), games in grouped_evaluation_games(data.games)
        if phase == "evaluation"
    ]
    final = [
        (step, summarize_games(games))
        for (phase, step), games in grouped_evaluation_games(data.games)
        if phase == "final_evaluation"
    ]
    score_axis = axes[0, 0]
    if periodic:
        score_axis.plot(
            [step for step, _ in periodic],
            [stats.score_rate * 100 for _, stats in periodic],
            marker="o",
            linewidth=2,
            label="Periodic evaluation",
        )
    if final:
        score_axis.scatter(
            [step for step, _ in final],
            [stats.score_rate * 100 for _, stats in final],
            marker="*",
            s=130,
            color="#d62728",
            label="Final evaluation",
            zorder=3,
        )
    score_axis.set_title("Evaluation score rate")
    score_axis.set_xlabel("Timestep")
    score_axis.set_ylabel("Score rate (%)")
    score_axis.set_ylim(0, 100)
    score_axis.grid(alpha=0.25)
    if periodic or final:
        score_axis.legend()

    stability_axis = axes[0, 1]
    plot_metric(stability_axis, data.metrics, "approx_kl", label="Approx KL")
    plot_metric(stability_axis, data.metrics, "clip_fraction", label="Clip fraction")
    stability_axis.set_title("PPO update stability")
    stability_axis.set_xlabel("Timestep")
    stability_axis.grid(alpha=0.25)
    stability_axis.legend()

    value_axis = axes[1, 0]
    plot_metric(value_axis, data.metrics, "value_loss", label="Value loss")
    explained_axis = value_axis.twinx()
    plot_metric(
        explained_axis,
        data.metrics,
        "explained_variance",
        label="Explained variance",
        color="#d62728",
    )
    explained_axis.axhline(0, color="#777777", linewidth=1, alpha=0.6)
    value_axis.set_title("Critic learning")
    value_axis.set_xlabel("Timestep")
    value_axis.set_ylabel("Value loss")
    explained_axis.set_ylabel("Explained variance")
    value_axis.grid(alpha=0.25)
    combine_legends(value_axis, explained_axis)

    entropy_axis = axes[1, 1]
    plot_metric(entropy_axis, data.metrics, "entropy", label="Entropy")
    entropy_axis.set_title("Policy entropy")
    entropy_axis.set_xlabel("Timestep")
    entropy_axis.set_ylabel("Entropy")
    entropy_axis.grid(alpha=0.25)

    if has_rollout_metrics:
        rollout_game_axis = axes[2, 0]
        plot_metric(
            rollout_game_axis,
            data.metrics,
            "completed_games",
            label="Completed games",
            phase="rollout",
        )
        plot_metric(
            rollout_game_axis,
            data.metrics,
            "decisive_games",
            label="Decisive games",
            phase="rollout",
        )
        reward_rate_axis = rollout_game_axis.twinx()
        plot_metric(
            reward_rate_axis,
            data.metrics,
            "reward_signal_rate",
            label="Reward signal rate",
            color="#d62728",
            phase="rollout",
        )
        rollout_game_axis.set_title("Rollout game and reward signals")
        rollout_game_axis.set_xlabel("Timestep")
        rollout_game_axis.set_ylabel("Games per rollout")
        reward_rate_axis.set_ylabel("Non-zero reward fraction")
        rollout_game_axis.grid(alpha=0.25)
        combine_legends(rollout_game_axis, reward_rate_axis)

        rollout_spread_axis = axes[2, 1]
        plot_metric(
            rollout_spread_axis,
            data.metrics,
            "return_std",
            label="Return std",
            phase="rollout",
        )
        plot_metric(
            rollout_spread_axis,
            data.metrics,
            "value_prediction_std",
            label="Value prediction std",
            phase="rollout",
        )
        plot_metric(
            rollout_spread_axis,
            data.metrics,
            "advantage_std",
            label="Advantage std",
            phase="rollout",
        )
        rollout_spread_axis.set_title("Critic target spread")
        rollout_spread_axis.set_xlabel("Timestep")
        rollout_spread_axis.grid(alpha=0.25)
        rollout_handles, _ = rollout_spread_axis.get_legend_handles_labels()
        if rollout_handles:
            rollout_spread_axis.legend()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_game_outcomes(data: ExperimentData, output_path: str | Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    figure.suptitle(f"Game Outcomes - {data.directory.name}", fontsize=14)
    evaluation_groups = grouped_evaluation_games(data.games)

    labels: list[str] = []
    stats_rows: list[GameStats] = []
    for (phase, step), games in evaluation_groups:
        label = f"{step:,}" if phase == "evaluation" else f"final\n{step:,}"
        labels.append(label)
        stats_rows.append(summarize_games(games))

    outcome_axis = axes[0]
    if stats_rows:
        positions = np.arange(len(stats_rows))
        wins = np.array([stats.wins for stats in stats_rows])
        draws = np.array([stats.draws for stats in stats_rows])
        losses = np.array([stats.losses for stats in stats_rows])
        outcome_axis.bar(positions, wins, label="Wins", color="#2ca02c")
        outcome_axis.bar(
            positions,
            draws,
            bottom=wins,
            label="Draws",
            color="#9e9e9e",
        )
        outcome_axis.bar(
            positions,
            losses,
            bottom=wins + draws,
            label="Losses",
            color="#d62728",
        )
        outcome_axis.set_xticks(positions, labels)
        outcome_axis.legend()
    else:
        outcome_axis.text(0.5, 0.5, "No evaluation games", ha="center", va="center")
    outcome_axis.set_title("Evaluation W/D/L")
    outcome_axis.set_xlabel("Timestep")
    outcome_axis.set_ylabel("Games")
    outcome_axis.grid(axis="y", alpha=0.25)

    train_games = tuple(game for game in data.games if game.phase == "train")
    terminations = Counter(game.termination for game in train_games)
    termination_axis = axes[1]
    if terminations:
        ordered = sorted(terminations.items(), key=lambda item: item[1])
        termination_axis.barh(
            [name for name, _ in ordered],
            [count for _, count in ordered],
            color="#4c78a8",
        )
        for index, (_, count) in enumerate(ordered):
            termination_axis.text(count, index, f" {count}", va="center")
    else:
        termination_axis.text(0.5, 0.5, "No training games", ha="center", va="center")
    termination_axis.set_title("Training game terminations")
    termination_axis.set_xlabel("Games")
    termination_axis.grid(axis="x", alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def grouped_evaluation_games(
    games: Iterable[GameRow],
) -> list[tuple[tuple[str, int], tuple[GameRow, ...]]]:
    grouped: dict[tuple[str, int], list[GameRow]] = defaultdict(list)
    for game in games:
        if game.phase in ("evaluation", "final_evaluation"):
            grouped[(game.phase, game.step)].append(game)
    return [
        (key, tuple(grouped[key]))
        for key in sorted(
            grouped,
            key=lambda item: (item[1], 0 if item[0] == "evaluation" else 1),
        )
    ]


def summarize_games(games: Iterable[GameRow]) -> GameStats:
    rows = tuple(games)
    wins = sum(game.reward > 0 for game in rows)
    draws = sum(game.reward == 0 for game in rows)
    losses = sum(game.reward < 0 for game in rows)
    score_rate = (wins + 0.5 * draws) / len(rows) if rows else 0.0
    average_plies = mean(game.plies for game in rows) if rows else 0.0
    terminations = tuple(
        sorted(
            Counter(game.termination for game in rows).items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return GameStats(
        games=len(rows),
        wins=wins,
        draws=draws,
        losses=losses,
        score_rate=score_rate,
        average_plies=average_plies,
        terminations=terminations,
    )


def metric_values(
    metrics: Iterable[MetricRow],
    *,
    phase: str,
    metric: str,
) -> list[float]:
    return [
        row.value
        for row in metrics
        if row.phase == phase and row.metric == metric and math.isfinite(row.value)
    ]


def plot_metric(
    axis: Any,
    metrics: Iterable[MetricRow],
    metric: str,
    *,
    label: str,
    color: str | None = None,
    phase: str = "train_update",
) -> None:
    rows = [
        row
        for row in metrics
        if row.phase == phase and row.metric == metric
    ]
    if not rows:
        return
    axis.plot(
        [row.step for row in rows],
        [row.value for row in rows],
        label=label,
        linewidth=1.8,
        color=color,
    )


def combine_legends(first_axis: Any, second_axis: Any) -> None:
    first_handles, first_labels = first_axis.get_legend_handles_labels()
    second_handles, second_labels = second_axis.get_legend_handles_labels()
    if first_handles or second_handles:
        first_axis.legend(
            first_handles + second_handles,
            first_labels + second_labels,
            loc="best",
        )


def find_best_checkpoint(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    best_events = [
        event
        for event in events
        if event.get("event") == "checkpoint" and bool(event.get("is_best"))
    ]
    return best_events[-1] if best_events else None


def completed_timesteps(data: ExperimentData) -> int:
    candidates = [row.step for row in data.metrics]
    candidates.extend(game.step for game in data.games)
    summary = data.summary_document.get("summary", {})
    if isinstance(summary, dict) and "completed_timesteps" in summary:
        candidates.append(int(summary["completed_timesteps"]))
    return max(candidates, default=0)


def experiment_duration(data: ExperimentData) -> str:
    timestamps = [
        str(data.config_document.get("created_at", "")),
        *(row.timestamp for row in data.metrics),
        *(row.timestamp for row in data.games),
    ]
    parsed = [parse_timestamp(value) for value in timestamps if value]
    parsed = [value for value in parsed if value is not None]
    if len(parsed) < 2:
        return "unknown"
    seconds = max(0, round((max(parsed) - min(parsed)).total_seconds()))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds}s"


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def read_metrics(path: Path) -> tuple[MetricRow, ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(
            MetricRow(
                timestamp=row["timestamp"],
                step=int(row["step"]),
                phase=row["phase"],
                metric=row["metric"],
                value=float(row["value"]),
            )
            for row in csv.DictReader(source)
        )


def read_games(path: Path) -> tuple[GameRow, ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(
            GameRow(
                timestamp=row["timestamp"],
                step=int(row["step"]),
                phase=row["phase"],
                episode=int(row["episode"]),
                result=row["result"],
                reward=float(row["reward"]),
                plies=int(row["plies"]),
                agent_color=row["agent_color"],
                opponent=row["opponent"],
                termination=row["termination"],
                checkpoint=row["checkpoint"],
            )
            for row in csv.DictReader(source)
        )


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def read_json_lines(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    events = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        event = json.loads(raw_line)
        if not isinstance(event, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        events.append(event)
    return tuple(events)


def format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_path",
        type=Path,
        help="one experiment directory or a parent whose experiments are all reported",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    results = generate_experiment_reports(
        args.experiment_path,
        output_dir=args.output_dir,
        create_plots=not args.no_plots,
    )
    for index, result in enumerate(results):
        if index:
            print()
        print(f"Experiment:      {result.experiment_dir}")
        print(f"Text summary:    {result.summary_path}")
        if result.learning_curves_path is not None:
            print(f"Learning curves: {result.learning_curves_path}")
        if result.game_outcomes_path is not None:
            print(f"Game outcomes:   {result.game_outcomes_path}")
    if len(results) > 1:
        print(f"\nGenerated reports: {len(results)}")


if __name__ == "__main__":
    main()
