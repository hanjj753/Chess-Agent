from pathlib import Path

import chess

from chess_agent.agents.base import Agent
from chess_agent.match import color_result, play_game, run_match


class FirstLegalAgent(Agent):
    name = "first-legal"

    def select_move(self, board: chess.Board) -> chess.Move | None:
        return next(iter(board.legal_moves), None)


class ResigningAgent(Agent):
    name = "resigning"

    def select_move(self, board: chess.Board) -> chess.Move | None:
        return None


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
    assert summary.agent_nodes == 0
    assert summary.agent_table_hits == 0


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


def test_color_result_marks_agent_black_score_green() -> None:
    assert color_result("1-0", chess.BLACK) == "\033[31m1\033[0m-\033[32m0\033[0m"


def test_color_result_marks_agent_white_score_green() -> None:
    assert color_result("0-1", chess.WHITE) == "\033[32m0\033[0m-\033[31m1\033[0m"


def test_color_result_can_be_disabled() -> None:
    assert color_result("1/2-1/2", chess.WHITE, use_color=False) == "1/2-1/2"


def test_play_game_saves_agent_loss_pgn(tmp_path: Path) -> None:
    summary = play_game(
        index=1,
        white_agent=ResigningAgent(),
        black_agent=FirstLegalAgent(),
        agent_color=chess.WHITE,
        fen=chess.STARTING_FEN,
        max_plies=10,
        white_name="alpha",
        black_name="stockfish",
        save_loss_dir=tmp_path,
    )

    assert summary.agent_score == 0.0
    assert summary.pgn_path is not None
    assert Path(summary.pgn_path).exists()
    assert '[White "alpha"]' in Path(summary.pgn_path).read_text(encoding="utf-8")


def test_play_game_does_not_save_draw_pgn(tmp_path: Path) -> None:
    summary = play_game(
        index=1,
        white_agent=FirstLegalAgent(),
        black_agent=FirstLegalAgent(),
        agent_color=chess.WHITE,
        fen=chess.STARTING_FEN,
        max_plies=1,
        save_loss_dir=tmp_path,
    )

    assert summary.agent_score == 0.5
    assert summary.pgn_path is None
    assert list(tmp_path.iterdir()) == []
