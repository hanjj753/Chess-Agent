from pathlib import Path

import pytest

import chess_agent.agents.uci_engine_agent as uci
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


def test_resolve_engine_path_finds_relative_engine_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci, "IS_WINDOWS", True)
    engines_dir = tmp_path / "engines"
    engine_dir = engines_dir / "stockfish"
    engine_dir.mkdir(parents=True)
    engine = engine_dir / "stockfish.exe"
    engine.write_text("", encoding="utf-8")

    assert resolve_engine_path("stockfish", engines_dir=engines_dir) == engine


def test_posix_folder_ignores_exe_weights_and_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci, "IS_WINDOWS", False)
    native = tmp_path / "stockfish-linux"
    for name in ("stockfish.exe", "README.md", "weights.nnue", native.name):
        (tmp_path / name).write_bytes(b"test")
    monkeypatch.setattr(uci.os, "access", lambda path, mode: Path(path).name in {native.name, "stockfish.exe"})

    assert find_engine_executable(tmp_path) == native


def test_posix_rejects_windows_only_folder_and_direct_exe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci, "IS_WINDOWS", False)
    exe = tmp_path / "stockfish.exe"
    exe.write_bytes(b"test")
    with pytest.raises(ValueError, match="No native Linux/macOS engine"):
        find_engine_executable(tmp_path)
    with pytest.raises(ValueError, match="Windows engine cannot run"):
        resolve_engine_path(exe)


def test_posix_missing_execute_permission_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci, "IS_WINDOWS", False)
    engine = tmp_path / "stockfish-linux"
    engine.write_bytes(b"test")
    monkeypatch.setattr(uci.os, "access", lambda path, mode: False)
    with pytest.raises(PermissionError, match="chmod"):
        resolve_engine_path(engine)
    with pytest.raises(FileNotFoundError, match="chmod"):
        find_engine_executable(tmp_path)


def test_relative_executable_path_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci, "IS_WINDOWS", False)
    monkeypatch.chdir(tmp_path)
    engine = tmp_path / "stockfish-linux"
    engine.write_bytes(b"test")
    monkeypatch.setattr(uci.os, "access", lambda path, mode: True)
    assert resolve_engine_path("./stockfish-linux", engines_dir=tmp_path / "engines") == engine


@pytest.mark.parametrize("windows", (True, False))
def test_multiple_executables_require_an_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: bool,
) -> None:
    monkeypatch.setattr(uci, "IS_WINDOWS", windows)
    suffix = ".exe" if windows else ""
    for name in ("stockfish-avx2", "stockfish-bmi2"):
        (tmp_path / (name + suffix)).write_bytes(b"test")
    monkeypatch.setattr(uci.os, "access", lambda path, mode: True)
    with pytest.raises(ValueError, match="Multiple engine executables"):
        find_engine_executable(tmp_path)


def test_find_engine_executable_reports_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_engine_executable(tmp_path)
