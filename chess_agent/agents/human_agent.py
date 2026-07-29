from collections.abc import Callable

import chess

from chess_agent.agents.base import Agent


class HumanAgent(Agent):
    name = "human"

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self.input_fn = input_fn
        self.output_fn = output_fn

    def select_move(self, board: chess.Board) -> chess.Move | None:
        if board.is_game_over(claim_draw=True):
            return None

        color = "White" if board.turn == chess.WHITE else "Black"

        while True:
            try:
                raw_move = self.input_fn(f"{color} move> ").strip()
            except EOFError:
                return None

            if raw_move.lower() in {"quit", "exit", "resign"}:
                return None

            move = parse_human_move(board, raw_move)
            if move is not None:
                return move

            self.output_fn(
                "Illegal move. Try SAN like 'Nf3' or UCI like 'g1f3'."
            )


def parse_human_move(board: chess.Board, raw_move: str) -> chess.Move | None:
    if not raw_move:
        return None

    try:
        return board.parse_san(raw_move)
    except ValueError:
        pass

    try:
        move = chess.Move.from_uci(raw_move)
    except ValueError:
        return None

    if move in board.legal_moves:
        return move

    return None
