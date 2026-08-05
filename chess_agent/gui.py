import tkinter as tk
from tkinter import font

import chess

from chess_agent.agents.base import Agent
from chess_agent.agents.human_agent import HumanAgent

SQUARE_SIZE = 72
BOARD_SIZE = SQUARE_SIZE * 8
LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"
SELECTED_SQUARE = "#f6f669"
LEGAL_TARGET = "#8fbf6a"
LAST_MOVE = "#cdd26a"

PIECE_SYMBOLS = {
    chess.Piece(chess.KING, chess.WHITE): "\u2654",
    chess.Piece(chess.QUEEN, chess.WHITE): "\u2655",
    chess.Piece(chess.ROOK, chess.WHITE): "\u2656",
    chess.Piece(chess.BISHOP, chess.WHITE): "\u2657",
    chess.Piece(chess.KNIGHT, chess.WHITE): "\u2658",
    chess.Piece(chess.PAWN, chess.WHITE): "\u2659",
    chess.Piece(chess.KING, chess.BLACK): "\u265A",
    chess.Piece(chess.QUEEN, chess.BLACK): "\u265B",
    chess.Piece(chess.ROOK, chess.BLACK): "\u265C",
    chess.Piece(chess.BISHOP, chess.BLACK): "\u265D",
    chess.Piece(chess.KNIGHT, chess.BLACK): "\u265E",
    chess.Piece(chess.PAWN, chess.BLACK): "\u265F",
}


def display_square(row: int, col: int, orientation: chess.Color) -> chess.Square:
    """Return the chess square shown at a GUI row and column."""
    if orientation == chess.WHITE:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def square_position(square: chess.Square, orientation: chess.Color) -> tuple[int, int]:
    """Return the GUI row and column for a chess square."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    if orientation == chess.WHITE:
        return 7 - rank_index, file_index
    return rank_index, 7 - file_index


def move_from_squares(
    board: chess.Board,
    from_square: chess.Square,
    to_square: chess.Square,
) -> chess.Move | None:
    """Return a legal move matching two clicked squares."""
    candidates = [
        move
        for move in board.legal_moves
        if move.from_square == from_square and move.to_square == to_square
    ]
    if not candidates:
        return None

    for move in candidates:
        if move.promotion == chess.QUEEN:
            return move
    return candidates[0]


def choose_orientation(agents: dict[chess.Color, Agent]) -> chess.Color:
    if isinstance(agents.get(chess.BLACK), HumanAgent):
        return chess.BLACK
    return chess.WHITE


class ChessGui:
    def __init__(
        self,
        *,
        board: chess.Board,
        agents: dict[chess.Color, Agent],
        max_plies: int,
        orientation: chess.Color | None = None,
        move_delay_ms: int = 300,
    ):
        self.board = board
        self.agents = agents
        self.max_plies = max_plies
        self.orientation = orientation if orientation is not None else choose_orientation(agents)
        self.move_delay_ms = move_delay_ms
        self.selected_square: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.game_stopped = False

        self.root = tk.Tk()
        self.root.title("Chess Agent")
        self.root.resizable(False, False)

        self.piece_font = font.Font(family="Segoe UI Symbol", size=38)
        self.coord_font = font.Font(family="Arial", size=9)

        self.canvas = tk.Canvas(
            self.root,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, rowspan=4)
        self.canvas.bind("<Button-1>", self.handle_click)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            width=34,
        )
        self.status_label.grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 6))

        self.move_list = tk.Listbox(self.root, width=34, height=24)
        self.move_list.grid(row=1, column=1, sticky="nsew", padx=12)

        self.flip_button = tk.Button(self.root, text="Flip Board", command=self.flip_board)
        self.flip_button.grid(row=2, column=1, sticky="ew", padx=12, pady=6)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_game)
        self.stop_button.grid(row=3, column=1, sticky="ew", padx=12, pady=(0, 12))

    def run(self) -> None:
        self.redraw()
        self.root.after(self.move_delay_ms, self.advance_game)
        self.root.mainloop()

    def advance_game(self) -> None:
        if self.game_stopped:
            self.update_status("Game stopped.")
            return

        if self.board.is_game_over(claim_draw=True):
            self.update_status(f"Game over: {self.board.result(claim_draw=True)}")
            return

        if len(self.board.move_stack) >= self.max_plies:
            self.update_status("Max plies reached: draw.")
            return

        agent = self.agents[self.board.turn]
        if isinstance(agent, HumanAgent):
            self.update_status(f"{self.side_name(self.board.turn)} to move.")
            return

        self.update_status(f"{self.side_name(self.board.turn)} agent thinking...")
        self.root.update_idletasks()
        move = agent.select_move(self.board)

        if move is None:
            self.game_stopped = True
            self.update_status(f"{self.side_name(self.board.turn)} resigns.")
            return

        self.apply_move(move, agent.name)
        self.root.after(self.move_delay_ms, self.advance_game)

    def handle_click(self, event: tk.Event) -> None:
        if self.board.is_game_over(claim_draw=True) or self.game_stopped:
            return

        agent = self.agents[self.board.turn]
        if not isinstance(agent, HumanAgent):
            return

        row = event.y // SQUARE_SIZE
        col = event.x // SQUARE_SIZE
        if not (0 <= row < 8 and 0 <= col < 8):
            return

        clicked_square = display_square(row, col, self.orientation)
        clicked_piece = self.board.piece_at(clicked_square)

        if self.selected_square is None:
            if clicked_piece is not None and clicked_piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.redraw()
            return

        if clicked_square == self.selected_square:
            self.selected_square = None
            self.redraw()
            return

        move = move_from_squares(self.board, self.selected_square, clicked_square)
        if move is None:
            if clicked_piece is not None and clicked_piece.color == self.board.turn:
                self.selected_square = clicked_square
            else:
                self.selected_square = None
            self.redraw()
            return

        self.selected_square = None
        self.apply_move(move, "human")
        self.root.after(self.move_delay_ms, self.advance_game)

    def apply_move(self, move: chess.Move, agent_name: str) -> None:
        san = self.board.san(move)
        moving_color = self.board.turn
        self.board.push(move)
        self.last_move = move
        self.move_list.insert(
            tk.END,
            f"{len(self.board.move_stack):03d}. {self.side_name(moving_color)} {agent_name}: {san}",
        )
        self.move_list.yview_moveto(1.0)
        self.redraw()

        if self.board.is_game_over(claim_draw=True):
            self.update_status(f"Game over: {self.board.result(claim_draw=True)}")

    def redraw(self) -> None:
        self.canvas.delete("all")

        for row in range(8):
            for col in range(8):
                square = display_square(row, col, self.orientation)
                self.draw_square(row, col, square)
                self.draw_piece(row, col, square)

    def draw_square(self, row: int, col: int, square: chess.Square) -> None:
        x1 = col * SQUARE_SIZE
        y1 = row * SQUARE_SIZE
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE
        color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE

        if self.last_move is not None and square in {
            self.last_move.from_square,
            self.last_move.to_square,
        }:
            color = LAST_MOVE

        if square == self.selected_square:
            color = SELECTED_SQUARE

        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

        if self.selected_square is not None:
            move = move_from_squares(self.board, self.selected_square, square)
            if move is not None:
                cx = x1 + SQUARE_SIZE / 2
                cy = y1 + SQUARE_SIZE / 2
                self.canvas.create_oval(
                    cx - 7,
                    cy - 7,
                    cx + 7,
                    cy + 7,
                    fill=LEGAL_TARGET,
                    outline=LEGAL_TARGET,
                )

        if col == 0:
            self.canvas.create_text(
                x1 + 5,
                y1 + 5,
                text=str(chess.square_rank(square) + 1),
                anchor="nw",
                font=self.coord_font,
                fill="#333333",
            )

        if row == 7:
            self.canvas.create_text(
                x2 - 5,
                y2 - 5,
                text=chess.FILE_NAMES[chess.square_file(square)],
                anchor="se",
                font=self.coord_font,
                fill="#333333",
            )

    def draw_piece(self, row: int, col: int, square: chess.Square) -> None:
        piece = self.board.piece_at(square)
        if piece is None:
            return

        self.canvas.create_text(
            col * SQUARE_SIZE + SQUARE_SIZE / 2,
            row * SQUARE_SIZE + SQUARE_SIZE / 2,
            text=PIECE_SYMBOLS[piece],
            font=self.piece_font,
            fill="#111111",
        )

    def flip_board(self) -> None:
        self.orientation = not self.orientation
        self.redraw()

    def stop_game(self) -> None:
        self.game_stopped = True
        self.update_status("Game stopped.")

    def update_status(self, message: str) -> None:
        self.status_var.set(message)

    @staticmethod
    def side_name(color: chess.Color) -> str:
        return "White" if color == chess.WHITE else "Black"


def run_gui_game(
    *,
    board: chess.Board,
    agents: dict[chess.Color, Agent],
    max_plies: int,
    move_delay_ms: int = 300,
) -> None:
    ChessGui(
        board=board,
        agents=agents,
        max_plies=max_plies,
        move_delay_ms=move_delay_ms,
    ).run()
