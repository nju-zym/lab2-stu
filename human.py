from agent import Agent
import tkinter as tk
from tkinter import messagebox
import threading
import time


class Human(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.selected_move = None
        self.waiting_for_move = False
        self.root = None
        self.canvas = None
        self.status_label = None
        self.board_size = 0
        self.cell_size = 30
        self.margin = 40
        self.canvas_size = 0
        self.surrendered = False

    def create_gui(self, board):
        """创建GUI棋盘界面"""
        self.board_size = len(board)

        if self.board_size <= 9:
            self.cell_size = 40
        elif self.board_size <= 15:
            self.cell_size = 30
        else:
            self.cell_size = 25

        self.canvas_size = (self.board_size - 1) * self.cell_size + 2 * self.margin

        if self.root is None:
            self.root = tk.Tk()
            self.root.title(f"五子棋 - 人类玩家 ({self.board_size}x{self.board_size})")

            window_width = self.canvas_size + 40
            window_height = self.canvas_size + 40
            self.root.geometry(f"{window_width}x{window_height}")
            self.root.resizable(False, False)

            self.canvas = tk.Canvas(
                self.root,
                width=self.canvas_size,
                height=self.canvas_size,
                bg="burlywood",
                highlightthickness=2,
                highlightbackground="black",
            )
            self.canvas.pack(pady=20)

            self.canvas.bind("<Button-1>", self.on_canvas_click)

            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.draw_board(board)

    def draw_board(self, board):
        """绘制棋盘"""
        if self.canvas is None:
            return

        self.canvas.delete("all")

        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            self.canvas.create_line(
                x,
                self.margin,
                x,
                self.margin + (self.board_size - 1) * self.cell_size,
                fill="black",
                width=1,
            )
            y = self.margin + i * self.cell_size
            self.canvas.create_line(
                self.margin,
                y,
                self.margin + (self.board_size - 1) * self.cell_size,
                y,
                fill="black",
                width=1,
            )

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] != 0:
                    self.draw_stone(i, j, board[i][j])

        font_size = min(max(8, self.cell_size // 3), 12)
        for i in range(self.board_size):
            y = self.margin + i * self.cell_size
            self.canvas.create_text(
                20, y, text=str(i), font=("Arial", font_size), fill="black"
            )
            x = self.margin + i * self.cell_size
            self.canvas.create_text(
                x, 20, text=str(i), font=("Arial", font_size), fill="black"
            )

    def draw_stone(self, row, col, player):
        """在指定位置绘制棋子"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = max(self.cell_size // 2 - 3, 8)

        if player == 1:
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="white",
                outline="black",
                width=2,
            )
        else:
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="black",
                outline="gray",
                width=2,
            )

    def on_canvas_click(self, event):
        """处理画布点击事件"""
        if not self.waiting_for_move:
            return

        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            nearest_x = self.margin + col * self.cell_size
            nearest_y = self.margin + row * self.cell_size
            distance = ((event.x - nearest_x) ** 2 + (event.y - nearest_y) ** 2) ** 0.5

            if distance <= self.cell_size / 3:
                self.selected_move = (row, col)
                self.waiting_for_move = False

    def on_closing(self):
        """处理窗口关闭事件 - 关闭窗口视为认输"""
        if self.waiting_for_move:
            self.selected_move = None
            self.waiting_for_move = False
            self.surrendered = True
            print(f"玩家 {self.player} 关闭窗口，视为认输！")
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
            self.status_label = None

    def make_move(self, board):
        """
        在棋盘上下一步棋。

        @param board: 表示游戏棋盘的二维列表
        @return 表示移动位置的元组 (行, 列)，如果认输则返回None
        """
        if self.surrendered:
            return None

        self.create_gui(board)

        self.selected_move = None
        self.waiting_for_move = True

        while self.waiting_for_move and self.root:
            try:
                self.root.update()
                time.sleep(0.01)
            except tk.TclError:
                break

        if self.surrendered:
            return None

        if self.selected_move is None:
            empty_cells = [
                (i, j)
                for i in range(len(board))
                for j in range(len(board))
                if board[i][j] == 0
            ]
            if empty_cells:
                import random

                return random.choice(empty_cells)
            return None

        return self.selected_move
