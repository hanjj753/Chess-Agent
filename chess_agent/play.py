import argparse

import chess

from chess_agent.agents import AlphaBetaAgent, HumanAgent, RandomAgent
from chess_agent.agents.base import Agent


def make_agent(kind: str, depth: int) -> Agent:
    if kind == "random":
        return RandomAgent()
    if kind == "alpha":
        return AlphaBetaAgent(depth=depth)
    if kind == "human":
        return HumanAgent()
    raise ValueError(f"unknown agent: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--white",
        choices=["random", "alpha", "human"],
        default="alpha",
    )
    parser.add_argument(
        "--black",
        choices=["random", "alpha", "human"],
        default="random",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    board = chess.Board(args.fen)
    agents = {
        chess.WHITE: make_agent(args.white, args.depth),
        chess.BLACK: make_agent(args.black, args.depth),
    }

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


if __name__ == "__main__":
    main()
