from pathlib import Path

import chess
import pytest

from chess_agent.analysis import MoveAnalysis, classify_loss
from chess_agent.batch_analysis import (
    GameAnalysis,
    MoveReference,
    analyze_or_load_game,
    average_loss_cp,
    capped_average_loss_cp,
    find_pgn_files,
    infer_agent_color,
    phase_summaries,
    references_for_games,
    summarize_games,
)


def sample_analysis(loss_cp: int = 80) -> MoveAnalysis:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    fen_before = board.fen()
    san = board.san(move)
    board.push(move)
    return MoveAnalysis(
        ply=1,
        move_number=1,
        color="white",
        san=san,
        uci=move.uci(),
        fen_before=fen_before,
        fen_after=board.fen(),
        score_before_cp=30,
        score_after_cp=30 - loss_cp,
        loss_cp=loss_cp,
        best_move_uci="g1f3",
        best_move_san="Nf3",
        label=classify_loss(loss_cp),
    )


def test_average_loss_cp_handles_empty_list() -> None:
    assert average_loss_cp([]) == 0.0


def test_capped_average_loss_cp_limits_mate_swings() -> None:
    assert capped_average_loss_cp([sample_analysis(100), sample_analysis(100_000)]) == 550


def test_find_pgn_files_returns_sorted_pgns(tmp_path: Path) -> None:
    (tmp_path / "b.pgn").write_text("", encoding="utf-8")
    (tmp_path / "a.pgn").write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    assert [path.name for path in find_pgn_files(tmp_path)] == ["a.pgn", "b.pgn"]


def test_find_pgn_files_rejects_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_pgn_files(tmp_path / "missing")


def test_analyze_or_load_game_uses_cached_json(tmp_path: Path) -> None:
    pgn_path = tmp_path / "loss_20260806_001419_game_006_black_1-0.pgn"
    pgn_path.write_text("", encoding="utf-8")
    analysis_path = pgn_path.with_suffix(".pgn.analysis.json")
    analysis_path.write_text(
        "["
        "{"
        '"ply": 1, "move_number": 1, "color": "white", "san": "e4", '
        '"uci": "e2e4", "fen_before": "fen1", "fen_after": "fen2", '
        '"score_before_cp": 0, "score_after_cp": -100, "loss_cp": 100, '
        '"best_move_uci": "g1f3", "best_move_san": "Nf3", '
        '"label": "inaccuracy"'
        "}"
        "]",
        encoding="utf-8",
    )

    game = analyze_or_load_game(
        pgn_path=pgn_path,
        engine_path=None,
        time_limit=0.1,
        depth=None,
        nodes=None,
        options=None,
        reuse_existing=True,
    )

    assert game.analysis_path == analysis_path
    assert game.agent_color == "black"
    assert game.moves[0].loss_cp == 100


def test_analyze_or_load_game_requires_engine_without_cache(tmp_path: Path) -> None:
    pgn_path = tmp_path / "loss.pgn"
    pgn_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        analyze_or_load_game(
            pgn_path=pgn_path,
            engine_path=None,
            time_limit=0.1,
            depth=None,
            nodes=None,
            options=None,
            reuse_existing=True,
        )


def test_phase_summaries_count_labels() -> None:
    opening = sample_analysis(80)
    middlegame = sample_analysis(160)
    middlegame = type(middlegame)(
        **{**middlegame.__dict__, "move_number": 15, "label": classify_loss(160)}
    )
    endgame = sample_analysis(320)
    endgame = type(endgame)(
        **{**endgame.__dict__, "move_number": 35, "label": classify_loss(320)}
    )

    summaries = phase_summaries(
        [
            MoveReference(Path("a.pgn"), opening),
            MoveReference(Path("b.pgn"), middlegame),
            MoveReference(Path("c.pgn"), endgame),
        ]
    )

    assert summaries[0].inaccuracies == 1
    assert summaries[1].mistakes == 1
    assert summaries[2].blunders == 1
    assert summaries[2].mate_like_losses == 0


def test_summarize_games_collects_top_losses() -> None:
    low = sample_analysis(20)
    high = sample_analysis(300)
    summary = summarize_games(
        [
            GameAnalysis(Path("low.pgn"), Path("low.json"), [low]),
            GameAnalysis(Path("high.pgn"), Path("high.json"), [high]),
        ]
    )

    assert summary.total_moves == 2
    assert summary.blunders == 1
    assert summary.capped_average_loss_cp == 160
    assert summary.top_losses[0].pgn_path == Path("high.pgn")


def test_references_for_games_can_filter_agent_moves() -> None:
    white_move = sample_analysis(100)
    black_move = type(white_move)(**{**white_move.__dict__, "color": "black"})
    game = GameAnalysis(
        Path("loss_20260806_001419_game_006_black_1-0.pgn"),
        Path("analysis.json"),
        [white_move, black_move],
        agent_color="black",
    )

    references = references_for_games([game], agent_only=True)

    assert [reference.move.color for reference in references] == ["black"]


def test_infer_agent_color_from_loss_filename() -> None:
    assert infer_agent_color("loss_20260806_001419_game_006_black_1-0.pgn") == "black"
    assert infer_agent_color("loss_without_color.pgn") is None
