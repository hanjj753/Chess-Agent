from pathlib import Path

from chess_agent.rl.evaluate_full_chess_ppo import (
    format_full_chess_report,
    save_report,
)
from chess_agent.rl.train_full_chess_ppo import (
    FullChessEvaluationResult,
    FullChessGameEvaluation,
)


def test_full_chess_evaluation_report_can_be_saved(tmp_path: Path) -> None:
    result = FullChessEvaluationResult(
        games=(
            FullChessGameEvaluation(
                episode=1,
                result="1-0",
                reward=1.0,
                plies=63,
                agent_color="white",
                termination="checkmate",
                illegal_action=False,
            ),
            FullChessGameEvaluation(
                episode=2,
                result="1/2-1/2",
                reward=0.0,
                plies=300,
                agent_color="black",
                termination="max_plies",
                illegal_action=False,
            ),
        )
    )
    report = format_full_chess_report(
        model_path="tmp/model.zip",
        opponent="random",
        result=result,
    )
    output_path = tmp_path / "reports" / "full_chess.txt"

    save_report(output_path, report)

    saved = output_path.read_text(encoding="utf-8")
    assert "W/D/L:          1/1/0" in saved
    assert "Score rate:     75.0%" in saved
    assert "Color breakdown" in saved
    assert "checkmate" in saved
    assert "max_plies" in saved
