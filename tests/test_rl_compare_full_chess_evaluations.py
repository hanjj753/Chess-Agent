from pathlib import Path

from chess_agent.rl.compare_full_chess_evaluations import (
    compare_evaluation_csvs,
    save_comparison,
)
from chess_agent.rl.evaluate_full_chess_ppo import save_game_results_csv
from chess_agent.rl.train_full_chess_ppo import (
    FullChessEvaluationResult,
    FullChessGameEvaluation,
)


def test_compare_evaluation_csvs_pairs_games_by_seed_and_color(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    save_game_results_csv(
        first_path,
        result=evaluation_result((1.0, 0.0, -1.0)),
        base_seed=10,
    )
    save_game_results_csv(
        second_path,
        result=evaluation_result((1.0, 1.0, 0.0)),
        base_seed=10,
    )

    report = compare_evaluation_csvs(first_path, second_path)
    output_path = save_comparison(tmp_path / "comparison.txt", report)

    assert "Paired games: 3" in report
    assert "Score delta:       +33.33%" in report
    assert "Improved/same/worse: 2/1/0" in report
    assert output_path.read_text(encoding="utf-8") == report


def evaluation_result(rewards: tuple[float, ...]) -> FullChessEvaluationResult:
    games = []
    for index, reward in enumerate(rewards, start=1):
        games.append(
            FullChessGameEvaluation(
                episode=index,
                result="1-0" if reward > 0 else "0-1" if reward < 0 else "1/2-1/2",
                reward=reward,
                plies=20 + index,
                agent_color="white" if index % 2 == 1 else "black",
                termination="checkmate" if reward else "max_plies",
                illegal_action=False,
            )
        )
    return FullChessEvaluationResult(games=tuple(games))
