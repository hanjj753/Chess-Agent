import argparse
from pathlib import Path

from chess_agent.rl.mate_in_one_env import ChessMateInOneEnv
from chess_agent.rl.random_baseline import EvaluationResult, evaluate_random_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "policy"], default="random")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    env = ChessMateInOneEnv()
    if args.agent == "random":
        result = evaluate_random_baseline(
            env=env,
            episodes=args.episodes,
            seed=args.seed,
        )
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required for --agent policy")
        result = evaluate_saved_policy(
            env=env,
            model_path=args.model_path,
            episodes=args.episodes,
        )

    print_result(args.agent, result)


def evaluate_saved_policy(
    *,
    env: ChessMateInOneEnv,
    model_path: Path,
    episodes: int,
) -> EvaluationResult:
    from chess_agent.rl.train_mate_in_one import evaluate_policy, load_policy

    policy = load_policy(model_path)
    return evaluate_policy(policy=policy, env=env, episodes=episodes)


def print_result(agent_name: str, result: EvaluationResult) -> None:
    print("Mate-in-one evaluation")
    print(f"Agent:          {agent_name}")
    print(f"Episodes:       {result.episodes}")
    print(f"Successes:      {result.successes}")
    print(f"Success rate:   {result.success_rate:.1%}")
    print(f"Illegal moves:  {result.illegal_actions}")
    print(f"Average reward: {result.average_reward:.3f}")


if __name__ == "__main__":
    main()
