import argparse
from pathlib import Path

from chess_agent.analysis import (
    analyze_pgn,
    default_analysis_path,
    load_analysis_json,
    save_analysis_json,
    suspicious_moves,
)
from chess_agent.agents.uci_engine_agent import parse_engine_options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", nargs="?", help="PGN file to analyze.")
    parser.add_argument("--engine", help="Path or engines/ subfolder for a UCI engine.")
    parser.add_argument("--engine-time", type=float, default=0.1)
    parser.add_argument("--engine-depth", type=int)
    parser.add_argument("--engine-nodes", type=int)
    parser.add_argument(
        "--engine-option",
        action="append",
        help="UCI option for the engine, written as Name=value",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis-json", type=Path, help="Open an existing analysis JSON.")
    parser.add_argument("--threshold-cp", type=int, default=75)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    if args.analysis_json is not None:
        analyses = load_analysis_json(args.analysis_json)
        output_path = args.analysis_json
    else:
        if args.pgn is None:
            parser.error("pgn is required unless --analysis-json is provided")
        if args.engine is None:
            parser.error("--engine is required when analyzing a PGN")

        analyses = analyze_pgn(
            pgn_path=args.pgn,
            engine_path=args.engine,
            time_limit=args.engine_time,
            depth=args.engine_depth,
            nodes=args.engine_nodes,
            options=parse_engine_options(args.engine_option),
        )
        output_path = args.output or default_analysis_path(args.pgn)
        save_analysis_json(analyses, output_path)

    print_report(analyses, threshold_cp=args.threshold_cp)
    print()
    print(f"Analysis JSON: {output_path}")

    if args.gui:
        from chess_agent.analysis_gui import run_analysis_gui

        run_analysis_gui(analyses)


def print_report(analyses, *, threshold_cp: int) -> None:
    flagged = suspicious_moves(analyses, threshold_cp=threshold_cp)

    print("Suspicious moves")
    if not flagged:
        print(f"No moves lost at least {threshold_cp} cp.")
        return

    for item in flagged:
        best = item.best_move_san or item.best_move_uci or "-"
        print(
            f"{item.ply:03d}. {item.color:5s} {item.san:8s} | "
            f"loss={item.loss_cp:4d} cp | {item.label:10s} | best={best}"
        )


if __name__ == "__main__":
    main()
