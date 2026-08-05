from pathlib import Path

import chess
import pytest

from chess_agent.analysis import (
    MoveAnalysis,
    classify_loss,
    default_analysis_path,
    load_analysis_json,
    save_analysis_json,
    suspicious_moves,
)
from chess_agent.analysis_gui import board_for_analysis_index


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


def test_classify_loss() -> None:
    assert classify_loss(20) == "ok"
    assert classify_loss(75) == "inaccuracy"
    assert classify_loss(150) == "mistake"
    assert classify_loss(300) == "blunder"


def test_suspicious_moves_filters_by_threshold() -> None:
    analyses = [sample_analysis(20), sample_analysis(100)]

    assert suspicious_moves(analyses, threshold_cp=75) == [analyses[1]]


def test_analysis_json_round_trips(tmp_path: Path) -> None:
    analyses = [sample_analysis()]
    path = tmp_path / "game.analysis.json"

    save_analysis_json(analyses, path)

    assert load_analysis_json(path) == analyses


def test_default_analysis_path_appends_suffix() -> None:
    assert default_analysis_path("loss.pgn") == Path("loss.pgn.analysis.json")


def test_board_for_analysis_index_returns_before_and_after_positions() -> None:
    analysis = sample_analysis()

    before = board_for_analysis_index([analysis], 0)
    after = board_for_analysis_index([analysis], 1)

    assert before.fen() == analysis.fen_before
    assert after.fen() == analysis.fen_after


def test_board_for_analysis_index_rejects_invalid_index() -> None:
    with pytest.raises(ValueError):
        board_for_analysis_index([sample_analysis()], 2)
