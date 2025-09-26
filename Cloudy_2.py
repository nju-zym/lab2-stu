import heapq
from agent import Agent


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (priority, _, item) = heapq.heappop(self.heap)
        return item, priority

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class Search(Agent):
    def check(self, white_count, black_count):
        self_count = opponent_count = add_score = 0
        if self.player == 1:
            self_count = white_count
            opponent_count = black_count
        elif self.player == 2:
            self_count = black_count
            opponent_count = white_count
        if self_count == 0 and opponent_count != 0:
            if opponent_count == 1:
                add_score = 1
            elif opponent_count == 2:
                add_score = 10
            elif opponent_count == 3:
                add_score = 50
            elif opponent_count == 4:
                add_score = 500
        elif opponent_count == 0 and self_count != 0:
            if self_count == 1:
                add_score = 1
            elif self_count == 2:
                add_score = 10
            elif self_count == 3:
                add_score = 100
            elif self_count == 4:
                add_score = 10000
        else:
            add_score = 0
        return add_score

    def plus_col(self, i, j, k, score_board, board):
        white_count = black_count = 0
        if i-k >= 0 and i-k+5 <= len(board):
            for x in range(i-k, i-k+5):
                if board[x][j] == 1:
                    white_count += 1
                elif board[x][j] == 2:
                    black_count += 1
            add_score = self.check(white_count, black_count)
            if add_score > 0:
                for x in range(i-k, i-k+5):
                    if board[x][j] == 0:
                        score_board[x][j][1] += add_score

    def plus_row(self, i, j, k, score_board, board):
        white_count = black_count = 0
        if j-k >= 0 and j-k+5 <= len(board):
            for y in range(j-k, j-k+5):
                if board[i][y] == 1:
                    white_count += 1
                elif board[i][y] == 2:
                    black_count += 1
            add_score = self.check(white_count, black_count)
            if add_score > 0:
                for y in range(j-k, j-k+5):
                    if board[i][y] == 0:
                        score_board[i][y][1] += add_score

    def plus_diag(self, i, j, k, score_board, board):
        white_count = black_count = 0
        if i-k >= 0 and i-k+5 <= len(board) and j-k >= 0 and j-k+5 <= len(board):
            for n in range(-k, -k+5):
                if board[i+n][j+n] == 1:
                    white_count += 1
                elif board[i+n][j+n] == 2:
                    black_count += 1
            add_score = self.check(white_count, black_count)
            if add_score > 0:
                for n in range(-k, -k+5):
                    if board[i+n][j+n] == 0:
                        score_board[i+n][j+n][1] += add_score

    def plus_anti_diag(self, i, j, k, score_board, board):
        white_count = black_count = 0
        if i-k >= 0 and i-k+5 <= len(board) and j+k-5 >= -1 and j+k <= len(board)-1:
            for n in range(-k, -k+5):
                if board[i+n][j-n] == 1:
                    white_count += 1
                elif board[i+n][j-n] == 2:
                    black_count += 1
            add_score = self.check(white_count, black_count)
            if add_score > 0:
                for n in range(-k, -k+5):
                    if board[i+n][j-n] == 0:
                        score_board[i+n][j-n][1] += add_score

    def plus_score(self, i, j, score_board, board):
        for k in range(5):
            self.plus_col(i, j, k, score_board, board)
            self.plus_row(i, j, k, score_board, board)
            self.plus_diag(i, j, k, score_board, board)
            self.plus_anti_diag(i, j, k, score_board, board)
        return

    def foresee(self, empty_cells_1, board, times):
        if times == 1:
            return empty_cells_1.pop()[0]
        else:
            probable_moves = PriorityQueue()
            for time in range(times):
                next_board = board.copy()
                probable_move, self_score = empty_cells_1.pop()
                next_board[probable_move[0]][probable_move[1]] = self.player
                score_board = [
                    [[(i, j), 0] for j in range(len(next_board))]
                    for i in range(len(next_board))
                ]
                for i in range(len(next_board)):
                    for j in range(len(next_board)):
                        if next_board[i][j] != 0:
                            self.plus_score(i, j, score_board, next_board)
                empty_cells_2 = PriorityQueue()
                for i in range(len(next_board)):
                    for j in range(len(next_board)):
                        if next_board[i][j] == 0:
                            empty_cells_2.update((i, j), -score_board[i][j][1])
                opponent_probable_move, opponent_score = empty_cells_2.pop()
                probable_moves.update(probable_move, self_score-opponent_score)
            choose_move = probable_moves.pop()[0]
            return choose_move

    @staticmethod
    def print_score_board(score_board):
        for i in range(len(score_board)):
            for j in range(len(score_board)):
                print(score_board[i][j][1], end="     ")
                if j == len(score_board)-1:
                    print("\n\n")

    def make_move(self, board):
        mid = len(board) // 2
        if board[mid][mid] == 0:
            return mid, mid
        else:
            score_board = [
                [[(i, j), 0] for j in range(len(board))]
                for i in range(len(board))
            ]
            for i in range(len(board)):
                for j in range(len(board)):
                    if board[i][j] != 0:
                        self.plus_score(i, j, score_board, board)
            empty_cells = PriorityQueue()
            for i in range(len(board)):
                for j in range(len(board)):
                    if board[i][j] == 0:
                        empty_cells.update((i, j), -score_board[i][j][1])
            move = (0, 0)
            match len(empty_cells.heap):
                case 1:
                    move = self.foresee(empty_cells, board, 1)
                case 2:
                    move = self.foresee(empty_cells, board, 2)
                case _:
                    move = self.foresee(empty_cells, board, 3)
            return move