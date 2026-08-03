import chess

from chess_agent.agents.base import Agent
from chess_agent.match import play_game, run_match


class FirstLegalAgent(Agent):
    name = "first-legal"

    def select_move(self, board: chess.Board) -> chess.Move | None:
        return next(iter(board.legal_moves), None)


def test_play_game_treats_max_plies_as_draw() -> None:
    summary = play_game(
        index=1,
        white_agent=FirstLegalAgent(),
        black_agent=FirstLegalAgent(),
        agent_color=chess.WHITE,
        fen=chess.STARTING_FEN,
        max_plies=1,
    )

    assert summary.result == "1/2-1/2"
    assert summary.plies == 1
    assert summary.termination == "max plies"


def test_run_match_alternates_agent_color() -> None:
    summary = run_match(
        agent=FirstLegalAgent(),
        opponent=FirstLegalAgent(),
        games=2,
        agent_start_color=chess.WHITE,
        alternate_colors=True,
        fen=chess.STARTING_FEN,
        max_plies=1,
    )

    assert [game.agent_color for game in summary.games] == [
        chess.WHITE,
        chess.BLACK,
    ]


def test_match_summary_counts_draws() -> None:
    summary = run_match(
        agent=FirstLegalAgent(),
        opponent=FirstLegalAgent(),
        games=2,
        agent_start_color=chess.WHITE,
        alternate_colors=True,
        fen=chess.STARTING_FEN,
        max_plies=1,
    )

    assert summary.agent_points == 1.0
    assert summary.opponent_points == 1.0
    assert summary.agent_wins == 0
    assert summary.draws == 2
    assert summary.opponent_wins == 0
