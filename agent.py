import random


class Agent:
    def __init__(self, player):
        self.player = player
        self.opponent = 3 - player

    def make_move(self, board):
        """
        在棋盘上下一步棋。

        @param board: 表示游戏棋盘的二维列表
        @return 表示移动位置的元组 (行, 列)
        """
        empty_cells = [
            (i, j)
            for i in range(len(board))
            for j in range(len(board))
            if board[i][j] == 0
        ]
        return random.choice(empty_cells) if empty_cells else None
