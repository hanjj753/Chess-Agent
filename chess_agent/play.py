import argparse

import chess

from chess_agent.agent_factory import AGENT_CHOICES, close_agent, make_agent
from chess_agent.agents.uci_engine_agent import parse_engine_options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--white",
        choices=AGENT_CHOICES,
        default="alpha",
    )
    parser.add_argument(
        "--black",
        choices=AGENT_CHOICES,
        default="random",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--white-engine", help="Path or engines/ subfolder for white UCI engine")
    parser.add_argument("--black-engine", help="Path or engines/ subfolder for black UCI engine")
    parser.add_argument("--engine-time", type=float, default=0.1)
    parser.add_argument("--engine-depth", type=int)
    parser.add_argument("--engine-nodes", type=int)
    parser.add_argument(
        "--white-engine-option",
        action="append",
        help="UCI option for white engine, written as Name=value",
    )
    parser.add_argument(
        "--black-engine-option",
        action="append",
        help="UCI option for black engine, written as Name=value",
    )
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    board = chess.Board(args.fen)
    agents = {
        chess.WHITE: make_agent(
            args.white,
            depth=args.depth,
            engine_path=args.white_engine,
            engine_time=args.engine_time,
            engine_depth=args.engine_depth,
            engine_nodes=args.engine_nodes,
            engine_options=parse_engine_options(args.white_engine_option),
        ),
        chess.BLACK: make_agent(
            args.black,
            depth=args.depth,
            engine_path=args.black_engine,
            engine_time=args.engine_time,
            engine_depth=args.engine_depth,
            engine_nodes=args.engine_nodes,
            engine_options=parse_engine_options(args.black_engine_option),
        ),
    }

    try:
        print(board)
        print()

        for ply in range(1, args.max_plies + 1):
            if board.is_game_over(claim_draw=True):
                break

            agent = agents[board.turn]
            move = agent.select_move(board)
            if move is None:
                break

            san = board.san(move)
            board.push(move)

            color = "White" if not board.turn else "Black"
            print(f"{ply:03d}. {color} {agent.name}: {san}")
            print(board)
            print()

        print("Result:", board.result(claim_draw=True))
    finally:
        for agent in agents.values():
            close_agent(agent)


if __name__ == "__main__":
    main()
