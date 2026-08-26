import argparse
from pathlib import Path

from chess_agent.rl.observations import history_observation_shape
from chess_agent.rl.policy import ConvolutionalPolicy
from chess_agent.rl.policy_value import (
    create_policy_value_from_policy,
    save_policy_value,
)
from chess_agent.rl.train_mate_in_one import load_policy


def initialize_policy_value(
    *,
    policy_path: str | Path,
    output_path: str | Path,
    history_length: int,
) -> Path:
    policy = load_policy(policy_path, device="cpu")
    if not isinstance(policy, ConvolutionalPolicy):
        raise ValueError("policy-value initialization requires a CNN policy checkpoint")

    input_channels = history_observation_shape(history_length)[0]
    model = create_policy_value_from_policy(
        policy,
        input_channels=input_channels,
    )
    return save_policy_value(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=4)
    args = parser.parse_args()

    output_path = initialize_policy_value(
        policy_path=args.policy_path,
        output_path=args.output_path,
        history_length=args.history_length,
    )
    input_channels = history_observation_shape(args.history_length)[0]
    print("Policy-value initialization")
    print(f"Source policy:  {args.policy_path}")
    print(f"History length: {args.history_length}")
    print(f"Input shape:    ({input_channels}, 8, 8)")
    print(f"Saved model:    {output_path}")


if __name__ == "__main__":
    main()
