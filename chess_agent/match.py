import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import time

import chess
import chess.pgn

from chess_agent.agent_factory import AGENT_CHOICES, close_agent, make_agent
from chess_agent.agents.base import Agent
from chess_agent.agents.uci_engine_agent import parse_engine_options

AGENT_COLOR = "\033[32m"
OPPONENT_COLOR = "\033[31m"
RESET_COLOR = "\033[0m"


@dataclass(frozen=True)
class GameSummary:
    index: int
    result: str
    agent_color: chess.Color
    plies: int
    termination: str
    agent_nodes: int = 0
    agent_table_hits: int = 0
    pgn: str = ""
    pgn_path: str | None = None

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
    white_name: str = "white",
    black_name: str = "black",
    save_loss_dir: Path | None = None,
) -> GameSummary:
    board = chess.Board(fen)
    agents = {
        chess.WHITE: white_agent,
        chess.BLACK: black_agent,
    }
    termination = "normal"
    agent_nodes = 0
    agent_table_hits = 0

    for ply in range(1, max_plies + 1):
        if board.is_game_over(claim_draw=True):
            break

        moving_color = board.turn
        move = agents[moving_color].select_move(board)
        if moving_color == agent_color:
            result = getattr(agents[moving_color], "last_result", None)
            if result is not None:
                agent_nodes += result.nodes
                agent_table_hits += result.table_hits

        if move is None:
            termination = "resignation"
            return build_game_summary(
                board=board,
                index=index,
                result="0-1" if moving_color == chess.WHITE else "1-0",
                agent_color=agent_color,
                plies=ply - 1,
                termination=termination,
                agent_nodes=agent_nodes,
                agent_table_hits=agent_table_hits,
                white_name=white_name,
                black_name=black_name,
                save_loss_dir=save_loss_dir,
            )

        board.push(move)
    else:
        if not board.is_game_over(claim_draw=True):
            return build_game_summary(
                board=board,
                index=index,
                result="1/2-1/2",
                agent_color=agent_color,
                plies=max_plies,
                termination="max plies",
                agent_nodes=agent_nodes,
                agent_table_hits=agent_table_hits,
                white_name=white_name,
                black_name=black_name,
                save_loss_dir=save_loss_dir,
            )

    return build_game_summary(
        board=board,
        index=index,
        result=board.result(claim_draw=True),
        agent_color=agent_color,
        plies=len(board.move_stack),
        termination=termination,
        agent_nodes=agent_nodes,
        agent_table_hits=agent_table_hits,
        white_name=white_name,
        black_name=black_name,
        save_loss_dir=save_loss_dir,
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
    agent_name: str = "agent",
    opponent_name: str = "opponent",
    save_loss_dir: Path | None = None,
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
            white_name = agent_name
            black_name = opponent_name
        else:
            white_agent = opponent
            black_agent = agent
            white_name = opponent_name
            black_name = agent_name

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
                white_name=white_name,
                black_name=black_name,
                save_loss_dir=save_loss_dir,
            )
        )

    return MatchSummary(game_summaries)


def print_match_summary(
    *,
    summary: MatchSummary,
    agent_name: str,
    opponent_name: str,
    use_color: bool = True,
) -> None:
    for game in summary.games:
        color = "white" if game.agent_color == chess.WHITE else "black"
        result = color_result(game.result, game.agent_color, use_color=use_color)
        print(
            f"Game {game.index:03d}: {result} | "
            f"agent={color:5s} | plies={game.plies:3d} | "
            f"nodes={game.agent_nodes:7d} | tt_hits={game.agent_table_hits:5d} | "
            f"{game.termination}"
        )
        if game.pgn_path is not None:
            print(f"           saved loss: {game.pgn_path}")

    total_games = len(summary.games)
    score_rate = summary.agent_points / total_games if total_games else 0.0

    print()
    print("Match summary")
    print(f"Agent:    {agent_name}")
    print(f"Opponent: {opponent_name}")
    print(f"Games:    {total_games}")
    print(
        "Score:    "
        f"{paint_role_score(summary.agent_points, AGENT_COLOR, use_color=use_color)} - "
        f"{paint_role_score(summary.opponent_points, OPPONENT_COLOR, use_color=use_color)} "
        f"({score_rate:.1%})"
    )
    print(
        "W/D/L:    "
        f"{summary.agent_wins}/{summary.draws}/{summary.opponent_wins}"
    )
    print(
        "Search:   "
        f"nodes={sum(game.agent_nodes for game in summary.games)} | "
        f"tt_hits={sum(game.agent_table_hits for game in summary.games)}"
    )


def color_result(
    result: str,
    agent_color: chess.Color,
    *,
    use_color: bool = True,
) -> str:
    if result == "1-0":
        white_score = paint_role_score(
            "1",
            AGENT_COLOR if agent_color == chess.WHITE else OPPONENT_COLOR,
            use_color=use_color,
        )
        black_score = paint_role_score(
            "0",
            AGENT_COLOR if agent_color == chess.BLACK else OPPONENT_COLOR,
            use_color=use_color,
        )
        return f"{white_score}-{black_score}"
    if result == "0-1":
        white_score = paint_role_score(
            "0",
            AGENT_COLOR if agent_color == chess.WHITE else OPPONENT_COLOR,
            use_color=use_color,
        )
        black_score = paint_role_score(
            "1",
            AGENT_COLOR if agent_color == chess.BLACK else OPPONENT_COLOR,
            use_color=use_color,
        )
        return f"{white_score}-{black_score}"
    if result == "1/2-1/2":
        white_score = paint_role_score(
            "1/2",
            AGENT_COLOR if agent_color == chess.WHITE else OPPONENT_COLOR,
            use_color=use_color,
        )
        black_score = paint_role_score(
            "1/2",
            AGENT_COLOR if agent_color == chess.BLACK else OPPONENT_COLOR,
            use_color=use_color,
        )
        return f"{white_score}-{black_score}"
    return result


def paint_role_score(
    score: float | str,
    color: str,
    *,
    use_color: bool = True,
) -> str:
    text = f"{score:.1f}" if isinstance(score, float) else score
    if not use_color:
        return text
    return f"{color}{text}{RESET_COLOR}"


def parse_color(raw_color: str) -> chess.Color:
    if raw_color == "white":
        return chess.WHITE
    if raw_color == "black":
        return chess.BLACK
    raise ValueError(f"unknown color: {raw_color}")


def build_game_summary(
    *,
    board: chess.Board,
    index: int,
    result: str,
    agent_color: chess.Color,
    plies: int,
    termination: str,
    agent_nodes: int,
    agent_table_hits: int,
    white_name: str,
    black_name: str,
    save_loss_dir: Path | None,
) -> GameSummary:
    pgn = board_to_pgn(
        board=board,
        result=result,
        white_name=white_name,
        black_name=black_name,
    )
    summary = GameSummary(
        index=index,
        result=result,
        agent_color=agent_color,
        plies=plies,
        termination=termination,
        agent_nodes=agent_nodes,
        agent_table_hits=agent_table_hits,
        pgn=pgn,
    )

    if save_loss_dir is None or summary.agent_score != 0.0:
        return summary

    pgn_path = save_loss_pgn(
        pgn=pgn,
        directory=save_loss_dir,
        index=index,
        result=result,
        agent_color=agent_color,
    )
    return GameSummary(
        index=index,
        result=result,
        agent_color=agent_color,
        plies=plies,
        termination=termination,
        agent_nodes=agent_nodes,
        agent_table_hits=agent_table_hits,
        pgn=pgn,
        pgn_path=str(pgn_path),
    )


def board_to_pgn(
    *,
    board: chess.Board,
    result: str,
    white_name: str,
    black_name: str,
) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Chess Agent Match"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["PlyCount"] = str(len(board.move_stack))
    return str(game) + "\n"


def save_loss_pgn(
    *,
    pgn: str,
    directory: Path,
    index: int,
    result: str,
    agent_color: chess.Color,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    color = "white" if agent_color == chess.WHITE else "black"
    safe_result = re.sub(r"[^A-Za-z0-9_.-]+", "_", result)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = directory / f"loss_{timestamp}_game_{index:03d}_{color}_{safe_result}.pgn"
    path.write_text(pgn, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="alpha")
    parser.add_argument("--opponent", choices=AGENT_CHOICES, default="random")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Seconds per move for local alpha-beta agents. Uses iterative deepening.",
    )
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
    parser.add_argument(
        "--save-losses",
        type=Path,
        help="Directory where PGNs for games lost by --agent are saved.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored score output.",
    )
    args = parser.parse_args()

    agent = make_agent(
        args.agent,
        depth=args.depth,
        time_limit=args.time_limit,
        engine_path=args.agent_engine,
        engine_time=args.engine_time,
        engine_depth=args.engine_depth,
        engine_nodes=args.engine_nodes,
        engine_options=parse_engine_options(args.agent_engine_option),
    )
    opponent = make_agent(
        args.opponent,
        depth=args.depth,
        time_limit=args.time_limit,
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
            agent_name=args.agent,
            opponent_name=args.opponent,
            save_loss_dir=args.save_losses,
        )
        print_match_summary(
            summary=summary,
            agent_name=args.agent,
            opponent_name=args.opponent,
            use_color=not args.no_color,
        )
    finally:
        close_agent(agent)
        close_agent(opponent)


if __name__ == "__main__":
    main()
