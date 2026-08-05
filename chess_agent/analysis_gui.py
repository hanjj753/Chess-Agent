import tkinter as tk
from tkinter import font

import chess

from chess_agent.analysis import MoveAnalysis
from chess_agent.gui import (
    BOARD_SIZE,
    DARK_SQUARE,
    LAST_MOVE,
    LIGHT_SQUARE,
    PIECE_SYMBOLS,
    SQUARE_SIZE,
    display_square,
)


class AnalysisReviewGui:
    def __init__(self, analyses: list[MoveAnalysis]):
        if not analyses:
            raise ValueError("analysis data must contain at least one move")

        self.analyses = analyses
        self.index = 0
        self.orientation = chess.WHITE
        self.root = tk.Tk()
        self.root.title("Chess Analysis Review")
        self.root.resizable(False, False)

        self.piece_font = font.Font(family="Segoe UI Symbol", size=38)
        self.coord_font = font.Font(family="Arial", size=9)

        self.canvas = tk.Canvas(
            self.root,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, rowspan=5)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            width=48,
        )
        self.status_label.grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 6))

        self.move_list = tk.Listbox(self.root, width=48, height=24)
        self.move_list.grid(row=1, column=1, sticky="nsew", padx=12)
        for item in analyses:
            self.move_list.insert(
                tk.END,
                f"{item.ply:03d}. {item.color:5s} {item.san:8s} "
                f"{item.loss_cp:4d} cp {item.label}",
            )
        self.move_list.bind("<<ListboxSelect>>", self.handle_select)

        controls = tk.Frame(self.root)
        controls.grid(row=2, column=1, sticky="ew", padx=12, pady=6)
        tk.Button(controls, text="Prev", command=self.prev_move).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(controls, text="Next", command=self.next_move).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(controls, text="Next Issue", command=self.next_issue).pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(self.root, text="Flip Board", command=self.flip_board).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=12,
            pady=6,
        )
        tk.Button(self.root, text="Close", command=self.root.destroy).grid(
            row=4,
            column=1,
            sticky="ew",
            padx=12,
            pady=(0, 12),
        )

    def run(self) -> None:
        self.redraw()
        self.root.mainloop()

    def current_board(self) -> chess.Board:
        return board_for_analysis_index(self.analyses, self.index)

    def current_move(self) -> MoveAnalysis | None:
        if self.index == 0:
            return None
        return self.analyses[self.index - 1]

    def redraw(self) -> None:
        board = self.current_board()
        current_move = self.current_move()
        self.canvas.delete("all")

        for row in range(8):
            for col in range(8):
                square = display_square(row, col, self.orientation)
                self.draw_square(row, col, square, current_move)
                self.draw_piece(row, col, square, board)

        self.update_status()
        self.move_list.selection_clear(0, tk.END)
        if current_move is not None:
            self.move_list.selection_set(self.index - 1)
            self.move_list.see(self.index - 1)

    def draw_square(
        self,
        row: int,
        col: int,
        square: chess.Square,
        current_move: MoveAnalysis | None,
    ) -> None:
        x1 = col * SQUARE_SIZE
        y1 = row * SQUARE_SIZE
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE
        color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE

        if current_move is not None:
            move = chess.Move.from_uci(current_move.uci)
            if square in {move.from_square, move.to_square}:
                color = LAST_MOVE

        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

    def draw_piece(self, row: int, col: int, square: chess.Square, board: chess.Board) -> None:
        piece = board.piece_at(square)
        if piece is None:
            return

        self.canvas.create_text(
            col * SQUARE_SIZE + SQUARE_SIZE / 2,
            row * SQUARE_SIZE + SQUARE_SIZE / 2,
            text=PIECE_SYMBOLS[piece],
            font=self.piece_font,
            fill="#111111",
        )

    def update_status(self) -> None:
        current_move = self.current_move()
        if current_move is None:
            self.status_var.set("Initial position")
            return

        best = current_move.best_move_san or current_move.best_move_uci or "-"
        self.status_var.set(
            f"Ply {current_move.ply}: {current_move.color} {current_move.san}\n"
            f"Loss: {current_move.loss_cp} cp ({current_move.label})\n"
            f"Score: {current_move.score_before_cp} -> {current_move.score_after_cp}\n"
            f"Engine preferred: {best}"
        )

    def handle_select(self, event: tk.Event) -> None:
        selection = self.move_list.curselection()
        if not selection:
            return
        self.index = selection[0] + 1
        self.redraw()

    def prev_move(self) -> None:
        self.index = max(0, self.index - 1)
        self.redraw()

    def next_move(self) -> None:
        self.index = min(len(self.analyses), self.index + 1)
        self.redraw()

    def next_issue(self) -> None:
        for next_index in range(self.index + 1, len(self.analyses) + 1):
            if self.analyses[next_index - 1].label != "ok":
                self.index = next_index
                self.redraw()
                return

    def flip_board(self) -> None:
        self.orientation = not self.orientation
        self.redraw()


def board_for_analysis_index(
    analyses: list[MoveAnalysis],
    index: int,
) -> chess.Board:
    if index < 0 or index > len(analyses):
        raise ValueError("analysis index out of range")
    if index == 0:
        return chess.Board(analyses[0].fen_before)
    return chess.Board(analyses[index - 1].fen_after)


def run_analysis_gui(analyses: list[MoveAnalysis]) -> None:
    AnalysisReviewGui(analyses).run()
