import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, cast

import chess
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from chess_agent.agents.alphabeta_agent import AlphaBetaAgent
from chess_agent.agents.alpha_random_agent import AlphaRandomAgent
from chess_agent.agents.base import Agent
from chess_agent.agents.random_agent import RandomAgent
from chess_agent.rl.experiment_tracking import ExperimentLogger
from chess_agent.rl.full_chess_env import BoardOnlyObservation, FullChessEnv
from chess_agent.rl.policy_value import PolicyValueNetwork, load_policy_value
from chess_agent.rl.ppo_policy import (
    ChessMaskableActorCriticPolicy,
    transfer_policy_value_to_ppo,
)


PPO_OPPONENTS = ("random", "alpha-random", "alpha")


@dataclass(frozen=True)
class FullChessPPOConfig:
    total_timesteps: int = 100_000
    additional_timesteps: int | None = None
    learning_rate: float = 3e-5
    n_envs: int = 4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float | None = 0.03
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    history_length: int = 4
    max_plies: int = 300
    reward_shaping_coefficient: float = 0.0
    reward_shaping_scale: float = 600.0
    hidden_size: int = 64
    dropout: float = 0.0
    residual_blocks: int = 3
    opponent: str = "random"
    alpha_move_probability: float = 0.1
    opponent_depth: int = 1
    opponent_time_limit: float | None = None
    evaluation_every: int = 10_000
    evaluation_games: int = 200
    checkpoint_every: int = 50_000
    deterministic_evaluation: bool = True
    initial_evaluation: bool = True
    seed: int = 0
    device: str = "auto"
    pretrained_policy_value_path: Path | None = None
    resume_from: Path | None = None
    save_path: Path = Path("tmp/full_chess_ppo_final.zip")
    checkpoint_dir: Path = Path("tmp/full_chess_ppo_checkpoints")
    initial_model_path: Path = Path("tmp/full_chess_ppo_initial.zip")
    best_model_path: Path = Path("tmp/full_chess_ppo_best.zip")
    experiment_dir: Path | None = Path("analysis/experiments")
    experiment_name: str = "full_chess_ppo"


@dataclass(frozen=True)
class FullChessGameEvaluation:
    episode: int
    result: str
    reward: float
    plies: int
    agent_color: str
    termination: str
    illegal_action: bool


@dataclass(frozen=True)
class FullChessEvaluationResult:
    games: tuple[FullChessGameEvaluation, ...]

    @property
    def episodes(self) -> int:
        return len(self.games)

    @property
    def wins(self) -> int:
        return sum(game.reward > 0 for game in self.games)

    @property
    def draws(self) -> int:
        return sum(game.reward == 0 for game in self.games)

    @property
    def losses(self) -> int:
        return sum(game.reward < 0 for game in self.games)

    @property
    def score_rate(self) -> float:
        if not self.games:
            return 0.0
        return (self.wins + 0.5 * self.draws) / len(self.games)

    @property
    def average_reward(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.reward for game in self.games) / len(self.games)

    @property
    def average_plies(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.plies for game in self.games) / len(self.games)

    @property
    def illegal_actions(self) -> int:
        return sum(game.illegal_action for game in self.games)


@dataclass(frozen=True)
class FullChessPPOTrainingResult:
    completed_timesteps: int
    start_timesteps: int
    target_timesteps: int
    trained_timesteps: int
    initial_model_path: Path
    final_model_path: Path
    best_model_path: Path
    experiment_run_dir: Path | None
    final_evaluation: FullChessEvaluationResult


class TrackedMaskablePPO(MaskablePPO):
    """MaskablePPO that mirrors SB3 training statistics to ExperimentLogger."""

    experiment_logger: ExperimentLogger | None = None

    def train(self) -> None:
        super().train()
        if self.experiment_logger is None:
            return

        source = self.logger.name_to_value
        metric_names = {
            "train/policy_gradient_loss": "policy_loss",
            "train/value_loss": "value_loss",
            "train/entropy_loss": "entropy_loss",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/explained_variance": "explained_variance",
            "train/loss": "total_loss",
            "train/learning_rate": "learning_rate",
            "train/n_updates": "updates",
        }
        metrics: dict[str, int | float] = {}
        for source_name, output_name in metric_names.items():
            raw_value = source.get(source_name)
            if raw_value is None:
                continue
            value = float(np.asarray(raw_value).item())
            if math.isfinite(value):
                metrics[output_name] = value
        entropy_loss = metrics.get("entropy_loss")
        if entropy_loss is not None:
            metrics["entropy"] = -float(entropy_loss)
        if metrics:
            self.experiment_logger.log_metrics(
                step=self.num_timesteps,
                phase="train_update",
                metrics=metrics,
            )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["experiment_logger"]


class FullChessTrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        config: FullChessPPOConfig,
        experiment_logger: ExperimentLogger | None,
    ) -> None:
        super().__init__(verbose=0)
        self.config = config
        self.experiment_logger = experiment_logger
        self.training_episode = 0
        self.best_score_rate = -1.0
        self.saved_best_this_run = False
        self.next_evaluation = 0
        self.next_checkpoint = 0
        self.rollout_game_rewards: list[float] = []
        self.rollout_extrinsic_rewards: list[float] = []
        self.rollout_shaping_rewards: list[float] = []

    def _on_training_start(self) -> None:
        self.next_evaluation = next_interval(
            self.num_timesteps,
            self.config.evaluation_every,
        )
        self.next_checkpoint = next_interval(
            self.num_timesteps,
            self.config.checkpoint_every,
        )

    def _on_rollout_start(self) -> None:
        self.rollout_game_rewards = []
        self.rollout_extrinsic_rewards = []
        self.rollout_shaping_rewards = []

    def _on_rollout_end(self) -> None:
        if self.experiment_logger is None:
            return

        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        rewards = np.asarray(rollout_buffer.rewards, dtype=np.float64)
        returns = np.asarray(rollout_buffer.returns, dtype=np.float64)
        values = np.asarray(rollout_buffer.values, dtype=np.float64)
        advantages = np.asarray(rollout_buffer.advantages, dtype=np.float64)
        transitions = int(rewards.size)
        completed_games = len(self.rollout_game_rewards)
        decisive_games = sum(
            not math.isclose(reward, 0.0)
            for reward in self.rollout_game_rewards
        )
        extrinsic_rewards = np.asarray(
            self.rollout_extrinsic_rewards,
            dtype=np.float64,
        )
        shaping_rewards = np.asarray(
            self.rollout_shaping_rewards,
            dtype=np.float64,
        )
        environment_training_rewards = extrinsic_rewards + shaping_rewards
        extrinsic_signal_rate = nonzero_rate(extrinsic_rewards)

        self.experiment_logger.log_metrics(
            step=self.num_timesteps,
            phase="rollout",
            metrics={
                "transitions": transitions,
                "completed_games": completed_games,
                "decisive_games": decisive_games,
                "reward_signal_rate": (
                    extrinsic_signal_rate
                ),
                "extrinsic_reward_signal_rate": extrinsic_signal_rate,
                "shaping_reward_signal_rate": nonzero_rate(shaping_rewards),
                "training_reward_signal_rate": nonzero_rate(
                    environment_training_rewards
                ),
                "mean_abs_shaping_reward": mean_absolute(shaping_rewards),
                "mean_abs_training_reward": mean_absolute(
                    environment_training_rewards
                ),
                "return_std": float(np.std(returns)),
                "value_prediction_std": float(np.std(values)),
                "advantage_std": float(np.std(advantages)),
            },
        )

    def _on_step(self) -> bool:
        self._record_step_rewards()
        self._record_completed_training_games()

        if self.config.checkpoint_every > 0:
            while self.num_timesteps >= self.next_checkpoint:
                checkpoint_path = self.config.checkpoint_dir / (
                    f"full_chess_ppo_{self.num_timesteps}.zip"
                )
                saved_path = save_ppo_model(self.model, checkpoint_path)
                if self.experiment_logger is not None:
                    self.experiment_logger.log_checkpoint(
                        step=self.num_timesteps,
                        path=saved_path,
                    )
                self.next_checkpoint += self.config.checkpoint_every

        if self.config.evaluation_every > 0:
            while self.num_timesteps >= self.next_evaluation:
                result = evaluate_full_chess_ppo_from_config(
                    model=cast(MaskablePPO, self.model),
                    config=self.config,
                    seed=self.config.seed + 100_000,
                )
                print_evaluation(self.num_timesteps, result)
                if self.experiment_logger is not None:
                    log_evaluation(
                        self.experiment_logger,
                        step=self.num_timesteps,
                        phase="evaluation",
                        opponent=self.config.opponent,
                        checkpoint=f"step:{self.num_timesteps}",
                        result=result,
                    )
                if result.score_rate > self.best_score_rate:
                    self.best_score_rate = result.score_rate
                    saved_path = save_ppo_model(self.model, self.config.best_model_path)
                    self.saved_best_this_run = True
                    if self.experiment_logger is not None:
                        self.experiment_logger.log_checkpoint(
                            step=self.num_timesteps,
                            path=saved_path,
                            is_best=True,
                            metrics={"score_rate": result.score_rate},
                        )
                self.next_evaluation += self.config.evaluation_every

        return True

    def _record_step_rewards(self) -> None:
        infos = self.locals.get("infos")
        if infos is None:
            return
        for info in infos:
            self.rollout_extrinsic_rewards.append(
                float(info.get("extrinsic_reward", 0.0))
            )
            self.rollout_shaping_rewards.append(
                float(info.get("shaping_reward", 0.0))
            )

    def _record_completed_training_games(self) -> None:
        if self.experiment_logger is None:
            return
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return

        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            self.training_episode += 1
            episode_info = info.get("episode", {})
            training_reward = float(
                info.get("episode_training_reward", episode_info.get("r", 0.0))
            )
            extrinsic_reward = float(info.get("episode_extrinsic_reward", 0.0))
            shaping_reward = float(info.get("episode_shaping_reward", 0.0))
            self.rollout_game_rewards.append(extrinsic_reward)
            self.experiment_logger.log_game(
                step=self.num_timesteps,
                phase="train",
                episode=self.training_episode,
                result=str(info.get("result", "*")),
                reward=extrinsic_reward,
                plies=int(info.get("episode_plies", 0)),
                agent_color=str(info.get("agent_color", "unknown")),
                opponent=self.config.opponent,
                termination=str(info.get("termination", "unknown")),
                extrinsic_reward=extrinsic_reward,
                shaping_reward=shaping_reward,
                training_reward=training_reward,
            )


def train_full_chess_ppo(
    config: FullChessPPOConfig,
) -> tuple[TrackedMaskablePPO, FullChessPPOTrainingResult]:
    validate_config(config)
    experiment_logger = (
        ExperimentLogger.create(
            config.experiment_dir,
            experiment_name=config.experiment_name,
            config=config,
        )
        if config.experiment_dir is not None
        else None
    )
    if experiment_logger is not None:
        print(f"Experiment log: {experiment_logger.run_dir}", flush=True)

    train_env = make_vector_env(config)
    pretrained_model: PolicyValueNetwork | None = None
    try:
        if config.resume_from is not None:
            model = TrackedMaskablePPO.load(
                config.resume_from,
                env=train_env,
                device=config.device,
            )
            model.target_kl = config.target_kl
        else:
            if config.pretrained_policy_value_path is not None:
                pretrained_model = load_policy_value(
                    config.pretrained_policy_value_path,
                    device="cpu",
                )
                hidden_size = pretrained_model.hidden_size
                dropout = config.dropout
                residual_blocks = pretrained_model.residual_blocks
            else:
                hidden_size = config.hidden_size
                dropout = config.dropout
                residual_blocks = config.residual_blocks

            model = TrackedMaskablePPO(
                ChessMaskableActorCriticPolicy,
                train_env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                target_kl=config.target_kl,
                ent_coef=config.entropy_coefficient,
                vf_coef=config.value_coefficient,
                max_grad_norm=config.max_grad_norm,
                policy_kwargs={
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "residual_blocks": residual_blocks,
                },
                verbose=1,
                seed=config.seed,
                device=config.device,
            )
            if pretrained_model is not None:
                transfer_policy_value_to_ppo(
                    source=pretrained_model,
                    target=cast(ChessMaskableActorCriticPolicy, model.policy),
                )

        model.experiment_logger = experiment_logger
        start_timesteps = model.num_timesteps
        target_timesteps = (
            start_timesteps + config.additional_timesteps
            if config.additional_timesteps is not None
            else config.total_timesteps
        )
        if start_timesteps > target_timesteps:
            raise ValueError(
                "resume checkpoint already exceeds the requested total_timesteps"
            )

        remaining_timesteps = target_timesteps - start_timesteps
        print(
            f"Training timesteps: start={start_timesteps:,} "
            f"additional={remaining_timesteps:,} target={target_timesteps:,}",
            flush=True,
        )
        initial_model_path = save_ppo_model(model, config.initial_model_path)
        if experiment_logger is not None:
            experiment_logger.log_checkpoint(
                step=model.num_timesteps,
                path=initial_model_path,
            )
        callback = FullChessTrainingCallback(
            config=config,
            experiment_logger=experiment_logger,
        )
        if config.initial_evaluation and config.evaluation_games > 0:
            initial_step = model.num_timesteps
            baseline = evaluate_full_chess_ppo_from_config(
                model=model,
                config=config,
                seed=config.seed + 100_000,
            )
            print_evaluation(initial_step, baseline)
            if experiment_logger is not None:
                log_evaluation(
                    experiment_logger,
                    step=initial_step,
                    phase="evaluation",
                    opponent=config.opponent,
                    checkpoint=f"step:{initial_step}:initial",
                    result=baseline,
                )
            callback.best_score_rate = baseline.score_rate
            baseline_path = save_ppo_model(model, config.best_model_path)
            callback.saved_best_this_run = True
            if experiment_logger is not None:
                experiment_logger.log_checkpoint(
                    step=initial_step,
                    path=baseline_path,
                    is_best=True,
                    metrics={"score_rate": baseline.score_rate},
                )
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )

        final_model_path = save_ppo_model(model, config.save_path)
        final_evaluation = evaluate_full_chess_ppo_from_config(
            model=model,
            config=config,
            seed=config.seed + 200_000,
        )
        best_model_path = normalized_ppo_path(config.best_model_path)
        if not callback.saved_best_this_run:
            callback.best_score_rate = final_evaluation.score_rate
            best_model_path = save_ppo_model(model, best_model_path)
            callback.saved_best_this_run = True
            if experiment_logger is not None:
                experiment_logger.log_checkpoint(
                    step=model.num_timesteps,
                    path=best_model_path,
                    is_best=True,
                    metrics={"score_rate": final_evaluation.score_rate},
                )
        if experiment_logger is not None:
            experiment_logger.log_checkpoint(
                step=model.num_timesteps,
                path=final_model_path,
            )
            log_evaluation(
                experiment_logger,
                step=model.num_timesteps,
                phase="final_evaluation",
                opponent=config.opponent,
                checkpoint=final_model_path,
                result=final_evaluation,
            )

        result = FullChessPPOTrainingResult(
            completed_timesteps=model.num_timesteps,
            start_timesteps=start_timesteps,
            target_timesteps=target_timesteps,
            trained_timesteps=model.num_timesteps - start_timesteps,
            initial_model_path=initial_model_path,
            final_model_path=final_model_path,
            best_model_path=best_model_path,
            experiment_run_dir=(
                experiment_logger.run_dir if experiment_logger is not None else None
            ),
            final_evaluation=final_evaluation,
        )
        if experiment_logger is not None:
            experiment_logger.save_summary(result)
        return model, result
    finally:
        train_env.close()


def evaluate_full_chess_ppo(
    *,
    model: MaskablePPO,
    episodes: int,
    history_length: int,
    max_plies: int,
    opponent: str,
    opponent_depth: int,
    opponent_time_limit: float | None,
    deterministic: bool,
    seed: int,
    alpha_move_probability: float = 0.1,
) -> FullChessEvaluationResult:
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    env = BoardOnlyObservation(
        FullChessEnv(
            opponent=make_opponent(
                opponent,
                alpha_move_probability=alpha_move_probability,
                depth=opponent_depth,
                time_limit=opponent_time_limit,
            ),
            history_length=history_length,
            max_plies=max_plies,
        )
    )
    games: list[FullChessGameEvaluation] = []
    try:
        for episode in range(1, episodes + 1):
            agent_color = chess.WHITE if episode % 2 == 1 else chess.BLACK
            observation, info = env.reset(
                seed=seed + episode - 1,
                options={"agent_color": agent_color},
            )
            episode_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=deterministic,
                    action_masks=env.action_masks(),
                )
                observation, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                episode_reward += float(reward)

            games.append(
                FullChessGameEvaluation(
                    episode=episode,
                    result=str(info.get("result", "*")),
                    reward=episode_reward,
                    plies=int(info.get("episode_plies", 0)),
                    agent_color=str(info.get("agent_color", "unknown")),
                    termination=str(info.get("termination", "unknown")),
                    illegal_action=bool(info.get("illegal_action", False)),
                )
            )
    finally:
        env.close()
    return FullChessEvaluationResult(games=tuple(games))


def evaluate_full_chess_ppo_from_config(
    *,
    model: MaskablePPO,
    config: FullChessPPOConfig,
    seed: int,
) -> FullChessEvaluationResult:
    return evaluate_full_chess_ppo(
        model=model,
        episodes=config.evaluation_games,
        history_length=config.history_length,
        max_plies=config.max_plies,
        opponent=config.opponent,
        alpha_move_probability=config.alpha_move_probability,
        opponent_depth=config.opponent_depth,
        opponent_time_limit=config.opponent_time_limit,
        deterministic=config.deterministic_evaluation,
        seed=seed,
    )


def make_vector_env(config: FullChessPPOConfig) -> VecEnv:
    env_fns: list[Callable[[], BoardOnlyObservation]] = []
    for rank in range(config.n_envs):
        env_fns.append(lambda rank=rank: make_single_env(config, rank=rank))
    return DummyVecEnv(env_fns)


def make_single_env(
    config: FullChessPPOConfig,
    *,
    rank: int,
) -> BoardOnlyObservation:
    env = FullChessEnv(
        opponent=make_opponent(
            config.opponent,
            alpha_move_probability=config.alpha_move_probability,
            depth=config.opponent_depth,
            time_limit=config.opponent_time_limit,
        ),
        history_length=config.history_length,
        max_plies=config.max_plies,
        reward_shaping_coefficient=config.reward_shaping_coefficient,
        reward_shaping_scale=config.reward_shaping_scale,
        reward_shaping_gamma=config.gamma,
    )
    env.reset(seed=config.seed + rank)
    return BoardOnlyObservation(Monitor(env))


def make_opponent(
    kind: str,
    *,
    alpha_move_probability: float = 0.1,
    depth: int,
    time_limit: float | None,
) -> Agent:
    if kind == "random":
        return RandomAgent()
    if kind == "alpha":
        return AlphaBetaAgent(depth=depth, time_limit=time_limit)
    if kind == "alpha-random":
        return AlphaRandomAgent(
            alpha_move_probability=alpha_move_probability,
            depth=depth,
            time_limit=time_limit,
        )
    raise ValueError(f"unsupported PPO opponent: {kind}")


def log_evaluation(
    logger: ExperimentLogger,
    *,
    step: int,
    phase: str,
    opponent: str,
    checkpoint: str | Path,
    result: FullChessEvaluationResult,
) -> None:
    logger.log_metrics(
        step=step,
        phase=phase,
        metrics={
            "episodes": result.episodes,
            "wins": result.wins,
            "draws": result.draws,
            "losses": result.losses,
            "score_rate": result.score_rate,
            "average_reward": result.average_reward,
            "average_plies": result.average_plies,
            "illegal_actions": result.illegal_actions,
        },
    )
    for game in result.games:
        logger.log_game(
            step=step,
            phase=phase,
            episode=game.episode,
            result=game.result,
            reward=game.reward,
            plies=game.plies,
            agent_color=game.agent_color,
            opponent=opponent,
            termination=game.termination,
            checkpoint=checkpoint,
        )


def print_evaluation(step: int, result: FullChessEvaluationResult) -> None:
    print(
        f"evaluation step={step} games={result.episodes} "
        f"W/D/L={result.wins}/{result.draws}/{result.losses} "
        f"score={result.score_rate:.1%} "
        f"avg_reward={result.average_reward:.3f} "
        f"avg_plies={result.average_plies:.1f}",
        flush=True,
    )


def save_ppo_model(model: MaskablePPO, path: str | Path) -> Path:
    output_path = normalized_ppo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return output_path


def normalized_ppo_path(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix.lower() != ".zip":
        output_path = Path(str(output_path) + ".zip")
    return output_path


def next_interval(current_step: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((current_step // interval) + 1) * interval


def nonzero_rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(values) > 1e-12) / values.size)


def mean_absolute(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def validate_config(config: FullChessPPOConfig) -> None:
    if config.total_timesteps < 0:
        raise ValueError("total_timesteps must be non-negative")
    if config.additional_timesteps is not None:
        if config.additional_timesteps < 0:
            raise ValueError("additional_timesteps must be non-negative")
        if config.resume_from is None:
            raise ValueError("additional_timesteps requires resume_from")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.n_envs < 1 or config.n_steps < 1:
        raise ValueError("n_envs and n_steps must be positive")
    rollout_size = config.n_envs * config.n_steps
    if rollout_size <= 1:
        raise ValueError("PPO rollout must contain more than one transition")
    if config.batch_size <= 1 or config.batch_size > rollout_size:
        raise ValueError("batch_size must be in [2, n_envs * n_steps]")
    if rollout_size % config.batch_size != 0:
        raise ValueError("batch_size must divide n_envs * n_steps")
    if config.n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    if config.target_kl is not None and (
        not math.isfinite(config.target_kl) or config.target_kl <= 0
    ):
        raise ValueError("target_kl must be positive or None")
    if config.dropout != 0:
        raise ValueError("PPO requires dropout=0 for stable probability ratios")
    if config.history_length < 0 or config.max_plies < 1:
        raise ValueError("invalid history_length or max_plies")
    if not math.isfinite(config.reward_shaping_coefficient) or (
        config.reward_shaping_coefficient < 0
    ):
        raise ValueError("reward_shaping_coefficient must be non-negative")
    if not math.isfinite(config.reward_shaping_scale) or (
        config.reward_shaping_scale <= 0
    ):
        raise ValueError("reward_shaping_scale must be positive")
    if config.evaluation_every < 0 or config.checkpoint_every < 0:
        raise ValueError("evaluation and checkpoint intervals must be non-negative")
    if config.evaluation_games < 0:
        raise ValueError("evaluation_games must be non-negative")
    if config.opponent not in PPO_OPPONENTS:
        raise ValueError(f"unsupported PPO opponent: {config.opponent}")
    if not math.isfinite(config.alpha_move_probability) or not (
        0.0 <= config.alpha_move_probability <= 1.0
    ):
        raise ValueError("alpha_move_probability must be between 0 and 1")
    if config.pretrained_policy_value_path is not None and config.resume_from is not None:
        raise ValueError("use pretrained_policy_value_path or resume_from, not both")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        help="timesteps to train beyond --resume-from instead of a cumulative target",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--no-target-kl",
        action="store_const",
        const=None,
        dest="target_kl",
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument(
        "--reward-shaping-coefficient",
        type=float,
        default=0.0,
        help="beta for potential-based static-evaluation reward shaping",
    )
    parser.add_argument(
        "--reward-shaping-scale",
        type=float,
        default=600.0,
        help="centipawn scale used to normalize the shaping potential",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--opponent", choices=PPO_OPPONENTS, default="random")
    parser.add_argument(
        "--alpha-move-probability",
        type=float,
        default=0.1,
        help="probability of an alpha move for the alpha-random opponent",
    )
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--opponent-time-limit", type=float)
    parser.add_argument("--evaluation-every", type=int, default=10_000)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--stochastic-evaluation", action="store_true")
    parser.add_argument("--no-initial-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained-policy-value", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--save-path", type=Path, default=Path("tmp/full_chess_ppo_final.zip"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("tmp/full_chess_ppo_checkpoints"),
    )
    parser.add_argument(
        "--initial-model-path",
        type=Path,
        default=Path("tmp/full_chess_ppo_initial.zip"),
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=Path("tmp/full_chess_ppo_best.zip"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("analysis/experiments"),
    )
    parser.add_argument("--experiment-name", default="full_chess_ppo")
    args = parser.parse_args()

    _, result = train_full_chess_ppo(
        FullChessPPOConfig(
            total_timesteps=args.total_timesteps,
            additional_timesteps=args.additional_timesteps,
            learning_rate=args.learning_rate,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            entropy_coefficient=args.entropy_coefficient,
            value_coefficient=args.value_coefficient,
            max_grad_norm=args.max_grad_norm,
            history_length=args.history_length,
            max_plies=args.max_plies,
            reward_shaping_coefficient=args.reward_shaping_coefficient,
            reward_shaping_scale=args.reward_shaping_scale,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            residual_blocks=args.residual_blocks,
            opponent=args.opponent,
            alpha_move_probability=args.alpha_move_probability,
            opponent_depth=args.opponent_depth,
            opponent_time_limit=args.opponent_time_limit,
            evaluation_every=args.evaluation_every,
            evaluation_games=args.evaluation_games,
            checkpoint_every=args.checkpoint_every,
            deterministic_evaluation=not args.stochastic_evaluation,
            initial_evaluation=not args.no_initial_evaluation,
            seed=args.seed,
            device=args.device,
            pretrained_policy_value_path=args.pretrained_policy_value,
            resume_from=args.resume_from,
            save_path=args.save_path,
            checkpoint_dir=args.checkpoint_dir,
            initial_model_path=args.initial_model_path,
            best_model_path=args.best_model_path,
            experiment_dir=args.experiment_dir,
            experiment_name=args.experiment_name,
        )
    )

    print()
    print("Full-chess PPO training summary")
    print(f"Start timesteps: {result.start_timesteps}")
    print(f"Added timesteps: {result.trained_timesteps}")
    print(f"Final timesteps: {result.completed_timesteps}")
    print(f"Initial model:   {result.initial_model_path}")
    print(f"Final model:     {result.final_model_path}")
    print(f"Best model:      {result.best_model_path}")
    print(f"Evaluation W/D/L: {result.final_evaluation.wins}/"
          f"{result.final_evaluation.draws}/{result.final_evaluation.losses}")
    print(f"Evaluation score: {result.final_evaluation.score_rate:.1%}")
    if result.experiment_run_dir is not None:
        print(f"Experiment log:  {result.experiment_run_dir}")


if __name__ == "__main__":
    main()
