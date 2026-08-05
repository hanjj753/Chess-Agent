import argparse
from pathlib import Path

from chess_agent.agents.uci_engine_agent import parse_engine_options
from chess_agent.batch_analysis import BatchSummary, analyze_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder containing PGN loss files.")
    parser.add_argument("--engine", help="Path or engines/ subfolder for a UCI engine.")
    parser.add_argument("--engine-time", type=float, default=0.1)
    parser.add_argument("--engine-depth", type=int)
    parser.add_argument("--engine-nodes", type=int)
    parser.add_argument(
        "--engine-option",
        action="append",
        help="UCI option for the engine, written as Name=value",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Re-analyze PGNs even if .analysis.json files already exist.",
    )
    parser.add_argument(
        "--all-moves",
        action="store_true",
        help="Include opponent moves too. By default only the saved-loss agent's moves are summarized.",
    )
    parser.add_argument("--loss-cap-cp", type=int, default=1000)
    args = parser.parse_args()

    summary = analyze_folder(
        folder=args.folder,
        engine_path=args.engine,
        time_limit=args.engine_time,
        depth=args.engine_depth,
        nodes=args.engine_nodes,
        options=parse_engine_options(args.engine_option),
        reuse_existing=not args.no_reuse,
        agent_only=not args.all_moves,
        loss_cap_cp=args.loss_cap_cp,
    )
    print_summary(summary)


def print_summary(summary: BatchSummary) -> None:
    print("Batch analysis summary")
    print(f"Scope:      {'agent moves only' if summary.agent_only else 'all moves'}")
    print(f"Games:      {len(summary.games)}")
    print(f"Moves:      {summary.total_moves}")
    print(f"Avg loss:   {summary.capped_average_loss_cp:.1f} cp capped")
    print(f"Raw avg:    {summary.average_loss_cp:.1f} cp")
    print(
        "Labels:     "
        f"inaccuracies={summary.inaccuracies}, "
        f"mistakes={summary.mistakes}, "
        f"blunders={summary.blunders}, "
        f"mate_like={summary.mate_like_losses}"
    )

    print()
    print("Phase breakdown")
    for phase in summary.phase_stats:
        print(
            f"{phase.name:10s} | moves={phase.move_count:4d} | "
            f"avg={phase.capped_average_loss_cp:6.1f} cp capped | "
            f"raw={phase.average_loss_cp:7.1f} | "
            f"I/M/B/Mate={phase.inaccuracies}/{phase.mistakes}/{phase.blunders}/{phase.mate_like_losses}"
        )

    print()
    print("Top losses")
    if not summary.top_losses:
        print("No analyzed moves.")
        return

    for index, reference in enumerate(summary.top_losses, start=1):
        move = reference.move
        best = move.best_move_san or move.best_move_uci or "-"
        print(
            f"{index:02d}. {reference.pgn_path.name} | "
            f"ply={move.ply:3d} | {move.color:5s} {move.san:8s} | "
            f"loss={move.loss_cp:4d} cp | {move.label:10s} | best={best}"
        )


if __name__ == "__main__":
    main()
