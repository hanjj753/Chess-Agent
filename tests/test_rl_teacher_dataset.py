from dataclasses import replace
import io
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import pytest

import chess_agent.rl.collect_teacher_dataset as collection
from chess_agent.rl.actions import ACTION_SIZE, legal_action_mask, move_to_action
from chess_agent.rl.observations import boards_to_history_observation
from chess_agent.rl.policy_value import PolicyValueNetwork, save_policy_value
from chess_agent.rl.teacher_dataset import (
    TeacherDataset,
    load_teacher_dataset,
    save_teacher_dataset,
)
from chess_agent.rl.ppo_policy import ChessMaskableActorCriticPolicy
from chess_agent.rl.train_full_chess_ppo import (
    FullChessPPOConfig,
    TrackedMaskablePPO,
    make_vector_env,
)


class FakeTeacher:
    options = {"Threads": None, "Hash": None, "UCI_LimitStrength": None}
    id = {"name": "test teacher"}

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[chess.Board, object, list | None]] = []
        self.configured: dict = {}

    def configure(self, options: dict) -> None:
        self.configured = options

    def analyse(self, board, limit, *, game, root_moves=None):
        self.calls.append((board.copy(stack=True), game, root_moves))
        move = root_moves[0] if root_moves else next(iter(board.legal_moves))
        return {
            "pv": [move],
            "score": chess.engine.PovScore(
                chess.engine.Cp(80 if root_moves else 120), board.turn,
            ),
        }

    def quit(self) -> None:
        self.closed = True


def sample_position(board: chess.Board | None = None) -> collection.PositionSample:
    board = board or chess.Board()
    moves = list(board.legal_moves)
    return collection.PositionSample(
        board=board.copy(stack=True),
        observation=boards_to_history_observation([board], history_length=1),
        action_mask=legal_action_mask(board),
        played_action=move_to_action(moves[-1]),
        ply=board.ply(),
    )


def labeled_dataset() -> TeacherDataset:
    return collection.label_game(
        samples=[sample_position()], engine=FakeTeacher(),
        limit=chess.engine.Limit(nodes=100), game_id=1,
        observation_shape=(36, 8, 8), metadata={"split": "train"},
    )


def test_teacher_dataset_round_trip_preserves_history_masks_and_labels(tmp_path: Path) -> None:
    original = labeled_dataset()
    path = save_teacher_dataset(tmp_path / "data.npz", original)
    dataset = load_teacher_dataset(path)

    assert len(dataset) == 1
    assert dataset.observation_shape == (36, 8, 8)
    assert dataset.metadata["split"] == "train"
    np.testing.assert_array_equal(dataset.observations(np.asarray([0]))[0], sample_position().observation)
    np.testing.assert_array_equal(dataset.action_masks(np.asarray([0]))[0], legal_action_mask(chess.Board()))
    assert dataset.columns["teacher_cp"][0] == 120
    assert dataset.columns["played_cp"][0] == 80
    assert dataset.cp_losses[0] == 40
    assert np.isnan(dataset.columns["teacher_mate"][0])
    assert not (tmp_path / "data.npz.tmp").exists()


@pytest.mark.parametrize("column", ("teacher_actions", "played_actions"))
def test_teacher_dataset_rejects_illegal_labels(tmp_path: Path, column: str) -> None:
    dataset = labeled_dataset()
    dataset.columns[column][0] = move_to_action(chess.Move.from_uci("a1a8"))
    with pytest.raises(ValueError, match="illegal action"):
        save_teacher_dataset(tmp_path / "bad.npz", dataset)


def test_teacher_dataset_rejects_ambiguous_scores(tmp_path: Path) -> None:
    dataset = labeled_dataset()
    dataset.columns["teacher_mate"][0] = 3
    with pytest.raises(ValueError, match="either cp or mate"):
        save_teacher_dataset(tmp_path / "bad.npz", dataset)


def test_labeling_keeps_move_stack_and_uses_same_root_perspective() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    sample = sample_position(board)
    engine = FakeTeacher()

    action, teacher_cp, teacher_mate, played_cp, _ = collection.label_position(
        engine, sample, chess.engine.Limit(nodes=100),
    )

    assert action == move_to_action(next(iter(board.legal_moves)))
    assert teacher_cp == 120
    assert played_cp == 80
    assert np.isnan(teacher_mate)
    assert len(engine.calls) == 2
    assert engine.calls[0][0].move_stack == board.move_stack
    assert engine.calls[0][1] is engine.calls[1][1]
    assert engine.calls[1][2] == [chess.Move.from_uci("a7a5")]
    assert board.fen() == sample.board.fen()


def test_agreed_move_reuses_teacher_score() -> None:
    sample = sample_position()
    sample = replace(sample, played_action=move_to_action(next(iter(sample.board.legal_moves))))
    engine = FakeTeacher()
    _, best_cp, _, played_cp, _ = collection.label_position(
        engine, sample, chess.engine.Limit(nodes=100),
    )
    assert len(engine.calls) == 1
    assert best_cp == played_cp == 120


def test_mate_scores_are_separate_and_not_averaged_as_cp() -> None:
    info = {"score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)}
    cp, mate = collection.score_columns(info, chess.BLACK)
    assert np.isnan(cp)
    assert mate == -3
    dataset = labeled_dataset()
    dataset.columns["teacher_cp"][0] = np.nan
    dataset.columns["teacher_mate"][0] = 3
    assert np.isnan(dataset.cp_losses[0])


def make_config(tmp_path: Path, **overrides) -> collection.TeacherCollectionConfig:
    model_path = tmp_path / "source.pt"
    save_policy_value(
        PolicyValueNetwork(input_channels=36, hidden_size=8, dropout=0.0, residual_blocks=1),
        model_path,
    )
    engine = tmp_path / "teacher"
    engine.write_text("test executable", encoding="utf-8")
    engine.chmod(0o755)
    return collection.TeacherCollectionConfig(**{
        "model_path": model_path, "engine_path": engine,
        "output_dir": tmp_path / "dataset", "games": 4,
        "positions_per_game": 4, "skip_plies": 0, "max_plies": 8,
        "opponent": "random", "validation_fraction": 0.5,
        "device": "cpu", "log_every": 0, "seed": 31,
        "engine_nodes": 100,
        "summary_output": tmp_path / "analysis" / "collection.txt",
        **overrides,
    })


def fake_engines(monkeypatch: pytest.MonkeyPatch) -> list[FakeTeacher]:
    engines: list[FakeTeacher] = []

    def popen(path):
        engine = FakeTeacher()
        engines.append(engine)
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", popen)
    return engines


def test_collection_splits_games_deduplicates_and_can_reuse_completed_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    engines = fake_engines(monkeypatch)
    summary = collection.collect_teacher_dataset(config)
    train = load_teacher_dataset(config.output_dir / "train.npz")
    valid = load_teacher_dataset(config.output_dir / "validation.npz")

    assert engines[0].closed
    assert engines[0].configured["UCI_LimitStrength"] is False
    assert collection.file_sha256(config.model_path) == train.metadata["manifest"]["model_sha256"]
    assert not set(train.columns["game_ids"]) & set(valid.columns["game_ids"])
    train_inputs = {row.tobytes() for row in train.columns["packed_observations"]}
    valid_inputs = {row.tobytes() for row in valid.columns["packed_observations"]}
    assert not train_inputs & valid_inputs
    assert len(train_inputs) == len(train)
    assert len(valid_inputs) == len(valid)
    assert summary["duplicates_removed"] >= 1
    assert len(train) + len(valid) + summary["duplicates_removed"] == 16
    assert "no policy or value weights were updated" in config.summary_output.read_text(encoding="utf-8")
    assert (config.output_dir / "positions.csv").is_file()
    pgn_text = (config.output_dir / "games.pgn").read_text(encoding="utf-8")
    source = io.StringIO(pgn_text)
    games = [chess.pgn.read_game(source) for _ in range(4)]
    assert all(game is not None and len(list(game.mainline_moves())) == 8 for game in games)
    games_by_id = {int(game.headers["GameId"]): game for game in games}
    for dataset in (train, valid):
        for row, game_id in enumerate(dataset.columns["game_ids"]):
            game = games_by_id[int(game_id)]
            board = game.board()
            history = [board.copy()]
            moves = list(game.mainline_moves())
            ply = int(dataset.columns["plies"][row])
            for move in moves[:ply]:
                board.push(move)
                history.append(board.copy())
            np.testing.assert_array_equal(
                dataset.observations(np.asarray([row]))[0],
                boards_to_history_observation(history, history_length=1),
            )
            assert dataset.columns["fens"][row] == board.fen()
            assert dataset.columns["played_actions"][row] == move_to_action(moves[ply])
    dates = [path.stat().st_mtime_ns for path in (config.output_dir / "games").glob("*.npz")]

    def unexpected_load(*args, **kwargs):
        raise AssertionError("completed resume should not load a model or run the teacher")

    monkeypatch.setattr(collection, "load_collection_policy", unexpected_load)
    resumed = collection.collect_teacher_dataset(replace(config, resume=True))
    assert resumed == summary
    assert dates == [path.stat().st_mtime_ns for path in (config.output_dir / "games").glob("*.npz")]
    assert len(engines) == 1


def test_collection_can_resume_after_an_interrupted_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    engines = fake_engines(monkeypatch)
    original = collection.label_game

    def interrupted(**kwargs):
        if kwargs["game_id"] == 2:
            raise KeyboardInterrupt()
        return original(**kwargs)

    monkeypatch.setattr(collection, "label_game", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collection.collect_teacher_dataset(config)
    first = collection.shard_path(config, 1)
    assert first.is_file()
    assert not collection.shard_path(config, 2).exists()
    assert engines[0].closed
    first_hash = collection.file_sha256(first)
    monkeypatch.setattr(collection, "label_game", original)
    result = collection.collect_teacher_dataset(replace(config, resume=True))
    assert result["games"] == 4
    assert collection.file_sha256(first) == first_hash
    assert engines[-1].closed


def test_collection_refuses_overwrite_and_incompatible_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    fake_engines(monkeypatch)
    collection.collect_teacher_dataset(config)
    with pytest.raises(FileExistsError, match="not empty"):
        collection.collect_teacher_dataset(config)
    with pytest.raises(ValueError, match="differs"):
        collection.collect_teacher_dataset(replace(config, resume=True, engine_nodes=200))
    config.model_path.write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="differs"):
        collection.collect_teacher_dataset(replace(config, resume=True))


def test_collection_accepts_a_ppo_zip_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, games=2, positions_per_game=1)
    env = make_vector_env(FullChessPPOConfig(
        n_envs=1, history_length=1, max_plies=8,
    ))
    path = tmp_path / "source.zip"
    try:
        model = TrackedMaskablePPO(
            ChessMaskableActorCriticPolicy, env,
            n_steps=2, batch_size=2, n_epochs=1, device="cpu",
            policy_kwargs={"hidden_size": 8, "dropout": 0.0, "residual_blocks": 1},
        )
        model.save(path)
    finally:
        env.close()
    fake_engines(monkeypatch)
    summary = collection.collect_teacher_dataset(replace(config, model_path=path))
    assert summary["train"]["positions"] == 1
    assert summary["validation"]["positions"] == 1
    assert summary["observation_shape"] == [36, 8, 8]


def test_teacher_missing_pv_or_score_fails_instead_of_writing_bad_labels() -> None:
    class BadTeacher(FakeTeacher):
        def analyse(self, board, limit, *, game, root_moves=None):
            return {}

    with pytest.raises(ValueError, match="principal variation"):
        collection.label_position(BadTeacher(), sample_position(), chess.engine.Limit(nodes=10))
    with pytest.raises(ValueError, match="no evaluation score"):
        collection.score_columns({}, chess.WHITE)


@pytest.mark.parametrize("overrides", (
    {"games": 1}, {"positions_per_game": 0}, {"skip_plies": 200},
    {"validation_fraction": float("nan")}, {"temperature": float("inf")},
    {"engine_time": 0}, {"engine_nodes": -1}, {"engine_threads": 0},
    {"engine_nodes": 100, "engine_depth": 2}, {"alpha_move_probability": 1.1},
))
def test_invalid_collection_config_is_rejected(tmp_path: Path, overrides: dict) -> None:
    config = collection.TeacherCollectionConfig(tmp_path / "model.zip", tmp_path / "data", **overrides)
    with pytest.raises(ValueError):
        collection.validate_config(config)


def test_engine_limit_uses_only_the_requested_budget(tmp_path: Path) -> None:
    config = collection.TeacherCollectionConfig(tmp_path / "model.zip", tmp_path / "data")
    assert collection.engine_limit(config).time == 0.05
    limit = collection.engine_limit(replace(config, engine_nodes=20000))
    assert limit.nodes == 20000
    assert limit.time is None
    assert limit.depth is None


def test_real_stockfish_collection_smoke(tmp_path: Path) -> None:
    import os

    engine = Path("engines/stockfish/stockfish-windows-x86-64-avx2.exe").resolve()
    if os.name != "nt" or not engine.exists():
        pytest.skip("local Windows Stockfish is not installed")
    config = make_config(tmp_path, engine_path=engine, games=2, positions_per_game=1)
    summary = collection.collect_teacher_dataset(config)
    assert summary["train"]["positions"] == 1
    assert summary["validation"]["positions"] == 1
    assert load_teacher_dataset(config.output_dir / "train.npz").action_masks(np.asarray([0])).shape == (1, ACTION_SIZE)
