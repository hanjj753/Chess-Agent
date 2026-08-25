from pathlib import Path

import chess
import numpy as np

from chess_agent.rl.actions import move_to_action
from chess_agent.rl.evaluate_tactical import (
    TacticalBreakdownAccumulator,
    evaluate_tactical_random_baseline,
    parse_episode_count,
    print_result,
    rating_bucket_label,
    rating_bucket_start,
    save_result_report,
)
from chess_agent.rl.tactical_puzzle_env import (
    TacticalPuzzle,
    TacticalPuzzleEnv,
    load_tactical_puzzles,
)


def test_tactical_env_plays_forced_line_to_success() -> None:
    env = TacticalPuzzleEnv(puzzles=[make_tactical_puzzle()])
    observation, info = env.reset(options={"puzzle_index": 0})

    assert observation["board"].shape == (18, 8, 8)
    assert info["expected_move_uci"] == "e7e5"

    observation, reward, terminated, _, info = env.step(action("e7e5"))

    assert reward == 0.0
    assert not terminated
    assert info["opponent_reply_uci"] == "g1f3"
    assert info["expected_move_uci"] == "b8c6"

    _, reward, terminated, _, info = env.step(action("b8c6"))

    assert reward == 1.0
    assert terminated
    assert info["is_success"]
    assert info["correct_agent_moves"] == 2


def test_tactical_env_rejects_wrong_legal_move() -> None:
    env = TacticalPuzzleEnv(puzzles=[make_tactical_puzzle()])
    env.reset(options={"puzzle_index": 0})

    _, reward, terminated, _, info = env.step(action("e7e6"))

    assert reward == -1.0
    assert terminated
    assert not info["is_correct"]
    assert info["expected_move_uci"] == "e7e5"


def test_tactical_env_action_masks_are_boolean_for_maskable_rl() -> None:
    env = TacticalPuzzleEnv(puzzles=[make_tactical_puzzle()])
    observation, _ = env.reset(options={"puzzle_index": 0})

    assert env.action_masks().dtype == np.bool_
    assert np.array_equal(env.action_masks(), observation["action_mask"].astype(bool))


def test_load_tactical_puzzles_reads_tsv(tmp_path: Path) -> None:
    path = tmp_path / "tactical.txt"
    puzzle = make_tactical_puzzle()
    path.write_text(
        f"{puzzle.initial_fen}\t{' '.join(puzzle.line_uci)}\t1200\tfork pin\n",
        encoding="utf-8",
    )

    puzzles = load_tactical_puzzles(path)

    assert len(puzzles) == 1
    assert puzzles[0].line_uci == ("e7e5", "g1f3", "b8c6")
    assert puzzles[0].rating == 1200
    assert puzzles[0].themes == ("fork", "pin")


def test_tactical_random_baseline_runs() -> None:
    result = evaluate_tactical_random_baseline(
        env=TacticalPuzzleEnv(puzzles=[make_tactical_puzzle()]),
        episodes=2,
        seed=0,
    )

    assert result.episodes == 2
    assert 0 <= result.move_accuracy <= 1


def test_tactical_evaluation_builds_metadata_breakdowns() -> None:
    first = make_tactical_puzzle()
    second = TacticalPuzzle(
        initial_fen=first.initial_fen,
        line_uci=first.line_uci,
        rating=1675,
        themes=("fork", "skewer"),
    )

    result = evaluate_tactical_random_baseline(
        env=TacticalPuzzleEnv(puzzles=[first, second]),
        episodes=2,
        seed=0,
    )

    ratings = {row.label: row for row in result.rating_breakdown}
    themes = {row.label: row for row in result.theme_breakdown}
    move_counts = {row.label: row for row in result.move_count_breakdown}

    assert ratings["1200-1399"].episodes == 1
    assert ratings["1600-1799"].episodes == 1
    assert themes["fork"].episodes == 2
    assert themes["pin"].episodes == 1
    assert themes["skewer"].episodes == 1
    assert move_counts["2"].episodes == 2
    assert {row.label for row in result.adjusted_theme_breakdown} == {
        "fork",
        "pin",
        "skewer",
    }


def test_adjusted_theme_breakdown_controls_rating_and_move_count() -> None:
    first = make_tactical_puzzle()
    second = TacticalPuzzle(
        initial_fen=first.initial_fen,
        line_uci=first.line_uci,
        rating=first.rating,
        themes=("skewer",),
    )
    accumulator = TacticalBreakdownAccumulator()
    accumulator.add(
        puzzle=first,
        success=1,
        correct_moves=2,
        expected_moves=2,
        total_reward=1.0,
    )
    accumulator.add(
        puzzle=second,
        success=0,
        correct_moves=0,
        expected_moves=2,
        total_reward=-1.0,
    )

    _, _, _, adjusted_rows = accumulator.freeze()
    adjusted = {row.label: row for row in adjusted_rows}

    assert adjusted["fork"].expected_success_rate == 0.5
    assert adjusted["fork"].success_gap == 0.5
    assert adjusted["skewer"].expected_success_rate == 0.5
    assert adjusted["skewer"].success_gap == -0.5


def test_print_tactical_result_filters_small_theme_groups(capsys) -> None:
    first = make_tactical_puzzle()
    second = TacticalPuzzle(
        initial_fen=first.initial_fen,
        line_uci=first.line_uci,
        rating=1675,
        themes=("fork", "skewer"),
    )
    result = evaluate_tactical_random_baseline(
        env=TacticalPuzzleEnv(puzzles=[first, second]),
        episodes=2,
        seed=0,
    )

    print_result("random", result, min_theme_episodes=2)
    output = capsys.readouterr().out

    assert "Rating breakdown" in output
    assert "Agent move-count breakdown" in output
    assert "Difficulty-adjusted theme breakdown" in output
    assert "fork" in output
    assert "pin" not in output
    assert "skewer" not in output


def test_rating_bucket_uses_two_hundred_point_ranges() -> None:
    assert rating_bucket_label(rating_bucket_start(1399)) == "1200-1399"
    assert rating_bucket_label(rating_bucket_start(1400)) == "1400-1599"
    assert rating_bucket_label(rating_bucket_start(None)) == "unknown"


def test_episode_count_accepts_all_or_integer() -> None:
    assert parse_episode_count("all") is None
    assert parse_episode_count("25") == 25


def test_tactical_result_can_be_saved_as_text(tmp_path: Path) -> None:
    result = evaluate_tactical_random_baseline(
        env=TacticalPuzzleEnv(puzzles=[make_tactical_puzzle()]),
        episodes=1,
        seed=0,
    )
    output_path = tmp_path / "reports" / "tactical.txt"

    saved_path = save_result_report(
        output_path,
        agent_name="random",
        result=result,
        min_theme_episodes=0,
    )

    report = output_path.read_text(encoding="utf-8")
    assert saved_path == output_path
    assert "Tactical puzzle evaluation" in report
    assert "Rating breakdown" in report
    assert "Difficulty-adjusted theme breakdown" in report
    assert "fork" in report


def make_tactical_puzzle() -> TacticalPuzzle:
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    return TacticalPuzzle(
        initial_fen=board.fen(),
        line_uci=("e7e5", "g1f3", "b8c6"),
        rating=1200,
        themes=("fork", "pin"),
    )


def action(move_uci: str) -> int:
    return move_to_action(chess.Move.from_uci(move_uci))
