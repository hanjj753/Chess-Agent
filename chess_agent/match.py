import argparse
from dataclasses import dataclass

import chess

from chess_agent.agent_factory import AGENT_CHOICES, close_agent, make_agent
from chess_agent.agents.base import Agent
from chess_agent.agents.uci_engine_agent import parse_engine_options


@dataclass(frozen=True)
class GameSummary:
    index: int
    result: str
    agent_color: chess.Color
    plies: int
    termination: str

    @property
    def agent_score(self) -> float:
        if self.result == "1/2-1/2":
            return 0.5
        if self.result == "1-0":
            return 1.0 if self.agent_color == chess.WHITE else 0.0
        if self.result == "0-1":
            return 1.0 if self.agent_color == chess.BLACK else 0.0
        return 0.0


@dataclass
class MatchSummary:
    games: list[GameSummary]

    @property
    def agent_points(self) -> float:
        return sum(game.agent_score for game in self.games)

    @property
    def opponent_points(self) -> float:
        return len(self.games) - self.agent_points

    @property
    def agent_wins(self) -> int:
        return sum(1 for game in self.games if game.agent_score == 1.0)

    @property
    def opponent_wins(self) -> int:
        return sum(1 for game in self.games if game.agent_score == 0.0)

    @property
    def draws(self) -> int:
        return sum(1 for game in self.games if game.agent_score == 0.5)


def play_game(
    *,
    index: int,
    white_agent: Agent,
    black_agent: Agent,
    agent_color: chess.Color,
    fen: str,
    max_plies: int,
) -> GameSummary:
    board = chess.Board(fen)
    agents = {
        chess.WHITE: white_agent,
        chess.BLACK: black_agent,
    }
    termination = "normal"

    for ply in range(1, max_plies + 1):
        if board.is_game_over(claim_draw=True):
            break

        moving_color = board.turn
        move = agents[moving_color].select_move(board)
        if move is None:
            termination = "resignation"
            return GameSummary(
                index=index,
                result="0-1" if moving_color == chess.WHITE else "1-0",
                agent_color=agent_color,
                plies=ply - 1,
                termination=termination,
            )

        board.push(move)
    else:
        if not board.is_game_over(claim_draw=True):
            return GameSummary(
                index=index,
                result="1/2-1/2",
                agent_color=agent_color,
                plies=max_plies,
                termination="max plies",
            )

    return GameSummary(
        index=index,
        result=board.result(claim_draw=True),
        agent_color=agent_color,
        plies=len(board.move_stack),
        termination=termination,
    )


def run_match(
    *,
    agent: Agent,
    opponent: Agent,
    games: int,
    agent_start_color: chess.Color,
    alternate_colors: bool,
    fen: str,
    max_plies: int,
    show_progress: bool = False,
) -> MatchSummary:
    game_summaries = []

    for game_index in range(1, games + 1):
        if alternate_colors and game_index % 2 == 0:
            agent_color = not agent_start_color
        else:
            agent_color = agent_start_color

        if agent_color == chess.WHITE:
            white_agent = agent
            black_agent = opponent
        else:
            white_agent = opponent
            black_agent = agent

        if show_progress:
            color = "white" if agent_color == chess.WHITE else "black"
            print(
                f"Running game {game_index}/{games} "
                f"(agent plays {color})...",
                flush=True,
            )

        game_summaries.append(
            play_game(
                index=game_index,
                white_agent=white_agent,
                black_agent=black_agent,
                agent_color=agent_color,
                fen=fen,
                max_plies=max_plies,
            )
        )

    return MatchSummary(game_summaries)


def print_match_summary(
    *,
    summary: MatchSummary,
    agent_name: str,
    opponent_name: str,
) -> None:
    for game in summary.games:
        color = "white" if game.agent_color == chess.WHITE else "black"
        print(
            f"Game {game.index:03d}: {game.result:7s} | "
            f"agent={color:5s} | plies={game.plies:3d} | {game.termination}"
        )

    total_games = len(summary.games)
    score_rate = summary.agent_points / total_games if total_games else 0.0

    print()
    print("Match summary")
    print(f"Agent:    {agent_name}")
    print(f"Opponent: {opponent_name}")
    print(f"Games:    {total_games}")
    print(
        "Score:    "
        f"{summary.agent_points:.1f} - {summary.opponent_points:.1f} "
        f"({score_rate:.1%})"
    )
    print(
        "W/D/L:    "
        f"{summary.agent_wins}/{summary.draws}/{summary.opponent_wins}"
    )


def parse_color(raw_color: str) -> chess.Color:
    if raw_color == "white":
        return chess.WHITE
    if raw_color == "black":
        return chess.BLACK
    raise ValueError(f"unknown color: {raw_color}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="alpha")
    parser.add_argument("--opponent", choices=AGENT_CHOICES, default="random")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--agent-start-color", choices=["white", "black"], default="white")
    parser.add_argument(
        "--fixed-colors",
        action="store_true",
        help="Do not alternate colors between games.",
    )
    parser.add_argument("--agent-engine", help="Path or engines/ subfolder for agent UCI engine")
    parser.add_argument("--opponent-engine", help="Path or engines/ subfolder for opponent UCI engine")
    parser.add_argument("--engine-time", type=float, default=0.1)
    parser.add_argument("--engine-depth", type=int)
    parser.add_argument("--engine-nodes", type=int)
    parser.add_argument(
        "--agent-engine-option",
        action="append",
        help="UCI option for agent engine, written as Name=value",
    )
    parser.add_argument(
        "--opponent-engine-option",
        action="append",
        help="UCI option for opponent engine, written as Name=value",
    )
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    agent = make_agent(
        args.agent,
        depth=args.depth,
        engine_path=args.agent_engine,
        engine_time=args.engine_time,
        engine_depth=args.engine_depth,
        engine_nodes=args.engine_nodes,
        engine_options=parse_engine_options(args.agent_engine_option),
    )
    opponent = make_agent(
        args.opponent,
        depth=args.depth,
        engine_path=args.opponent_engine,
        engine_time=args.engine_time,
        engine_depth=args.engine_depth,
        engine_nodes=args.engine_nodes,
        engine_options=parse_engine_options(args.opponent_engine_option),
    )

    try:
        summary = run_match(
            agent=agent,
            opponent=opponent,
            games=args.games,
            agent_start_color=parse_color(args.agent_start_color),
            alternate_colors=not args.fixed_colors,
            fen=args.fen,
            max_plies=args.max_plies,
            show_progress=True,
        )
        print_match_summary(
            summary=summary,
            agent_name=args.agent,
            opponent_name=args.opponent,
        )
    finally:
        close_agent(agent)
        close_agent(opponent)


if __name__ == "__main__":
    main()
