import numpy as np
import math
import numpy as np
import math
import time
from agent import Agent


class Search(Agent):
    """五子棋搜索AI (迭代加深 + AlphaBeta + 置换表 + Killer + History)
    仅依赖 numpy, 控制单步时间 <60s.
    """

    def __init__(self, player):
        super().__init__(player)
        self.max_time = 55.0
        self.start_time = None
        self.time_exceeded = False
        self.best_move = None
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.tt = {}
        self.killer_moves = {}
        self.history_score = {}

    # ---------------- 主入口 ----------------
    def make_move(self, board: np.ndarray):
        self.start_time = time.time()
        self.time_exceeded = False
        self.best_move = None
        self.tt.clear()
        self.killer_moves.clear()
        self.history_score.clear()

        candidates = self.generate_candidates(board)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        win_now = self.find_winning_move(board, self.player)
        if win_now:
            return win_now
        block = self.find_winning_move(board, self.opponent)
        if block:
            return block

        depth = 1
        last_move = None
        try:
            while True:
                mv = self.search_root(board, depth, candidates)
                if mv is not None:
                    last_move = mv
                if self.time_exceeded:
                    break
                cand_len = len(candidates)
                max_d = 5
                if cand_len > 30:
                    max_d = 3
                elif cand_len > 18:
                    max_d = 4
                if depth >= max_d:
                    break
                depth += 1
        except Exception:
            pass
        return last_move or self.best_move or candidates[0]

    # ------------- 根节点搜索 -------------
    def search_root(self, board, depth, candidates):
        prelim = []
        for mv in candidates:
            if self.out_of_time():
                break
            r, c = mv
            board[r][c] = self.player
            score = self.fast_evaluate(board)
            board[r][c] = 0
            neigh = self.count_neighbors(board, r, c)
            hist = self.history_score.get((self.player, mv), 0)
            prelim.append((score + neigh * 50 + hist * 0.5, mv))
        prelim.sort(reverse=True, key=lambda x: x[0])
        ordered = [m for _, m in prelim[:35]]
        alpha = -math.inf
        beta = math.inf
        best_val = -math.inf
        best_move = None
        for mv in ordered:
            if self.out_of_time():
                break
            r, c = mv
            board[r][c] = self.player
            val = self.alphabeta(board, depth - 1, False, alpha, beta, (r, c), ply=1)
            board[r][c] = 0
            if val > best_val:
                best_val = val
                best_move = mv
            alpha = max(alpha, best_val)
        if best_move:
            self.best_move = best_move
        return best_move

    # ------------- Alpha-Beta -------------
    def alphabeta(self, board, depth, maximizing, alpha, beta, last_move, ply):
        if self.out_of_time():
            return self.fast_evaluate(board)
        if depth == 0 or (last_move and self.is_win(board, last_move)):
            return self.evaluate(board)
        player_to_move = self.player if maximizing else self.opponent
        orig_alpha, orig_beta = alpha, beta
        key = (board.tobytes(), player_to_move)
        entry = self.tt.get(key)
        if entry and entry[0] >= depth:
            d, val, flag, a_s, b_s, mv_best = entry
            if flag == 0:
                return val
            if flag == -1 and val <= alpha:
                return val
            if flag == 1 and val >= beta:
                return val
        moves = self.generate_candidates(board)
        win = self.find_winning_move(board, player_to_move)
        if win:
            r, c = win
            board[r][c] = player_to_move
            v = self.evaluate(board)
            board[r][c] = 0
            return v
        block = self.find_winning_move(board, 3 - player_to_move)
        if block:
            if block in moves:
                moves.remove(block)
            moves.insert(0, block)
        if not moves:
            return self.evaluate(board)

        center = (len(board) - 1) / 2.0
        def move_key(m):
            r, c = m
            neigh = self.count_neighbors(board, r, c)
            dist = abs(r - center) + abs(c - center)
            hist = self.history_score.get((player_to_move, m), 0)
            killer_bonus = 1000 if m in self.killer_moves.get(ply, []) else 0
            return (neigh, -dist, hist, killer_bonus)
        if entry and entry[5] in moves:
            bm = entry[5]
            moves.remove(bm)
            moves.insert(0, bm)
        moves.sort(key=move_key, reverse=True)

        if maximizing:
            value = -math.inf
            best_local = None
            for (r, c) in moves:
                board[r][c] = self.player
                child = self.alphabeta(board, depth - 1, False, alpha, beta, (r, c), ply + 1)
                board[r][c] = 0
                if child > value:
                    value = child
                    best_local = (r, c)
                if value > alpha:
                    alpha = value
                if beta <= alpha:
                    km = self.killer_moves.setdefault(ply, [])
                    if best_local and best_local not in km:
                        km.insert(0, best_local)
                        if len(km) > 2:
                            km.pop()
                    self.history_score[(player_to_move, best_local)] = self.history_score.get((player_to_move, best_local), 0) + depth * depth
                    break
        else:
            value = math.inf
            best_local = None
            for (r, c) in moves:
                board[r][c] = self.opponent
                child = self.alphabeta(board, depth - 1, True, alpha, beta, (r, c), ply + 1)
                board[r][c] = 0
                if child < value:
                    value = child
                    best_local = (r, c)
                if value < beta:
                    beta = value
                if beta <= alpha:
                    km = self.killer_moves.setdefault(ply, [])
                    if best_local and best_local not in km:
                        km.insert(0, best_local)
                        if len(km) > 2:
                            km.pop()
                    self.history_score[(player_to_move, best_local)] = self.history_score.get((player_to_move, best_local), 0) + depth * depth
                    break
        flag = 0
        if value <= orig_alpha:
            flag = -1
        elif value >= orig_beta:
            flag = 1
        self.tt[key] = (depth, value, flag, alpha, beta, best_local)
        return value

    # ------------- 生成候选 -------------
    def generate_candidates(self, board):
        size = len(board)
        stones = np.argwhere(board != 0)
        if stones.size == 0:
            mid = size // 2
            return [(mid, mid)]
        occupied = set((int(r), int(c)) for r, c in stones)
        cand = set()
        for r, c in occupied:
            for dr in range(-2, 3):
                nr = r + dr
                if nr < 0 or nr >= size:
                    continue
                for dc in range(-2, 3):
                    nc = c + dc
                    if nc < 0 or nc >= size:
                        continue
                    if board[nr][nc] == 0:
                        cand.add((nr, nc))
        if not cand:
            for i in range(size):
                for j in range(size):
                    if board[i][j] == 0:
                        cand.add((i, j))
        lst = list(cand)
        lst.sort(key=lambda mv: self.count_neighbors(board, mv[0], mv[1]), reverse=True)
        return lst

    # ------------- 战术赢着 -------------
    def find_winning_move(self, board, player):
        for r, c in self.generate_candidates(board):
            board[r][c] = player
            if self.has_five(board, player):
                board[r][c] = 0
                return (r, c)
            board[r][c] = 0
        return None

    def count_neighbors(self, board, r, c):
        size = len(board)
        cnt = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] != 0:
                    cnt += 1
        return cnt

    def is_win(self, board, last_move):
        if not last_move:
            return False
        r, c = last_move
        player = board[r][c]
        if player == 0:
            return False
        size = len(board)
        for dx, dy in self.directions:
            count = 1
            x, y = r + dx, c + dy
            while 0 <= x < size and 0 <= y < size and board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = r - dx, c - dy
            while 0 <= x < size and 0 <= y < size and board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= 5:
                return True
        return False

    def evaluate(self, board):
        if self.has_five(board, self.player):
            return 1_000_000_000
        if self.has_five(board, self.opponent):
            return -1_000_000_000
        my_score = self.pattern_score(board, self.player)
        op_score = self.pattern_score(board, self.opponent)
        return my_score - op_score * 1.1

    def fast_evaluate(self, board):
        return self.pattern_score(board, self.player, light=True) - 1.05 * self.pattern_score(board, self.opponent, light=True)

    def has_five(self, board, player):
        size = len(board)
        for r in range(size):
            for c in range(size):
                if board[r][c] != player:
                    continue
                for dx, dy in self.directions:
                    cnt = 1
                    x, y = r + dx, c + dy
                    while 0 <= x < size and 0 <= y < size and board[x][y] == player:
                        cnt += 1
                        if cnt >= 5:
                            return True
                        x += dx
                        y += dy
        return False

    def pattern_score(self, board, player, light=False):
        size = len(board)
        score = 0
        if light:
            table = {4: (3000000, 200000), 3: (40000, 4000), 2: (500, 80), 1: (8, 1)}
        else:
            table = {4: (5_000_000, 300_000), 3: (80_000, 8_000), 2: (1_500, 200), 1: (10, 2)}
        for dx, dy in self.directions:
            for r in range(size):
                for c in range(size):
                    if board[r][c] != player:
                        continue
                    pr, pc = r - dx, c - dy
                    if 0 <= pr < size and 0 <= pc < size and board[pr][pc] == player:
                        continue
                    length = 0
                    x, y = r, c
                    while 0 <= x < size and 0 <= y < size and board[x][y] == player:
                        length += 1
                        x += dx
                        y += dy
                    if length >= 5:
                        score += 1_000_000_000
                        continue
                    if length > 4:
                        continue
                    lr, lc = r - dx, c - dy
                    left_blocked = not (0 <= lr < size and 0 <= lc < size) or board[lr][lc] not in (0, player)
                    rr, rc = x, y
                    right_blocked = not (0 <= rr < size and 0 <= rc < size) or board[rr][rc] not in (0, player)
                    open_ends = 2 - (1 if left_blocked else 0) - (1 if right_blocked else 0)
                    if length in table and open_ends > 0:
                        score += table[length][0] if open_ends == 2 else table[length][1]
        return score

    def out_of_time(self):
        if self.time_exceeded:
            return True
        if time.time() - self.start_time > self.max_time:
            self.time_exceeded = True
            return True
        return False


if __name__ == "__main__":
    size = 11
    b = np.zeros((size, size), dtype=int)
    ai = Search(1)
    print("Test first move:", ai.make_move(b))
