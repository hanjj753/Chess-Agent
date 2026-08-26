import argparse
from collections import Counter
from pathlib import Path

from chess_agent.rl.observations import OBSERVATION_CHANNELS
from chess_agent.rl.train_full_chess_ppo import (
    PPO_OPPONENTS,
    FullChessEvaluationResult,
    TrackedMaskablePPO,
    evaluate_full_chess_ppo,
)


def format_full_chess_report(
    *,
    model_path: str | Path,
    opponent: str,
    result: FullChessEvaluationResult,
) -> str:
    lines = [
        "Full-chess PPO evaluation",
        f"Model:          {model_path}",
        f"Opponent:       {opponent}",
        f"Games:          {result.episodes}",
        f"W/D/L:          {result.wins}/{result.draws}/{result.losses}",
        f"Score rate:     {result.score_rate:.1%}",
        f"Average reward: {result.average_reward:.3f}",
        f"Average plies:  {result.average_plies:.1f}",
        f"Illegal moves:  {result.illegal_actions}",
        "",
        "Color breakdown",
        "Color   Games   W/D/L   Score",
    ]
    for color in ("white", "black"):
        games = tuple(game for game in result.games if game.agent_color == color)
        wins = sum(game.reward > 0 for game in games)
        draws = sum(game.reward == 0 for game in games)
        losses = sum(game.reward < 0 for game in games)
        score_rate = (wins + 0.5 * draws) / len(games) if games else 0.0
        lines.append(
            f"{color:5s} {len(games):7d}   {wins}/{draws}/{losses}   {score_rate:6.1%}"
        )

    lines.extend(["", "Termination breakdown", "Termination             Games"])
    terminations = Counter(game.termination for game in result.games)
    for termination, games in sorted(
        terminations.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"{termination:22s} {games:5d}")
    return "\n".join(lines) + "\n"


def infer_history_length(model: TrackedMaskablePPO) -> int:
    shape = model.observation_space.shape
    if shape is None or len(shape) != 3 or shape[1:] != (8, 8):
        raise ValueError("model does not use a chess board observation")
    channels = int(shape[0])
    if channels % OBSERVATION_CHANNELS != 0:
        raise ValueError("model input channels are not a multiple of 18")
    return channels // OBSERVATION_CHANNELS - 1


def save_report(path: str | Path, report: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opponent", choices=PPO_OPPONENTS, default="random")
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--opponent-time-limit", type=float)
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    model = TrackedMaskablePPO.load(args.model_path, device=args.device)
    history_length = (
        args.history_length
        if args.history_length is not None
        else infer_history_length(model)
    )
    result = evaluate_full_chess_ppo(
        model=model,
        episodes=args.games,
        history_length=history_length,
        max_plies=args.max_plies,
        opponent=args.opponent,
        opponent_depth=args.opponent_depth,
        opponent_time_limit=args.opponent_time_limit,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    report = format_full_chess_report(
        model_path=args.model_path,
        opponent=args.opponent,
        result=result,
    )
    print(report, end="")
    if args.output_path is not None:
        saved_path = save_report(args.output_path, report)
        print(f"Saved report:   {saved_path}")


if __name__ == "__main__":
    main()
