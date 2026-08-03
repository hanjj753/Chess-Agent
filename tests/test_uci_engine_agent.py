from pathlib import Path

import pytest

from chess_agent.agents.uci_engine_agent import (
    find_engine_executable,
    parse_engine_option,
    parse_engine_options,
    resolve_engine_path,
)


def test_parse_engine_option_converts_common_values() -> None:
    assert parse_engine_option("Skill Level=5") == ("Skill Level", 5)
    assert parse_engine_option("UCI_LimitStrength=true") == (
        "UCI_LimitStrength",
        True,
    )
    assert parse_engine_option("Contempt=0.5") == ("Contempt", 0.5)
    assert parse_engine_option("WeightsFile=maia.pb.gz") == (
        "WeightsFile",
        "maia.pb.gz",
    )


def test_parse_engine_options_keeps_last_value() -> None:
    assert parse_engine_options(["Skill Level=1", "Skill Level=3"]) == {
        "Skill Level": 3,
    }


def test_resolve_engine_path_finds_relative_engine_folder(tmp_path: Path) -> None:
    engines_dir = tmp_path / "engines"
    engine_dir = engines_dir / "stockfish"
    engine_dir.mkdir(parents=True)
    engine = engine_dir / "stockfish.exe"
    engine.write_text("", encoding="utf-8")

    assert resolve_engine_path("stockfish", engines_dir=engines_dir) == engine


def test_find_engine_executable_reports_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_engine_executable(tmp_path)
