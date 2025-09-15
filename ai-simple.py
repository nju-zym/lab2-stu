import time, random
from agent import Agent

WIN = 1_000_000

PATTERN_SCORES = {
    "FIVE": WIN,
    "LIVE_FOUR": 50_000,  # .XXXX.
    "BROKEN_FOUR": 45_000,  # XXX.X / XX.XX （至少一端可延）
    "LIVE_THREE": 2_000,  # .XXX.
    "JUMP_THREE": 500,  # .XX.X. / .X.XX.
    "LIVE_TWO": 100,  # .XX.. / ..XX. / .X.X.
}

DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 纵、横、两斜


class SearchLite(Agent):  # 如需继承 Agent：class SearchLite(Agent):
    def __init__(self, player):
        super().__init__(player)  # 若继承自 Agent 别忘了
        self.tt = {}  # zobrist_hash -> (depth, score, best_move)
        self.hist = {}  # 简易历史启发 (r,c) -> int
        self.zkeys = None
        self.deadline = None
        self.N = None

    # ---------- 公共入口 ----------
    def make_move(self, board):
        tlim = 58.5
        self.N = len(board)
        if self.zkeys is None:
            self._init_zobrist(self.N)

        # 根部立即赢/堵
        my_win = self._find_immediate_win(board, self.player)
        if my_win:
            return my_win
        opp_win = self._find_immediate_win(board, self.opponent)
        if opp_win:
            return opp_win

        self.deadline = time.time() + tlim
        best_move = None
        for depth in range(1, self._max_depth(board) + 1):
            score, move = self._negamax(
                board, depth, -(10**9), 10**9, self.player, None, depth
            )
            if move is not None:
                best_move = move
            if time.time() > self.deadline:
                break
        if best_move is None:
            # 兜底：中心或第一空位
            c = self.N // 2
            return (c, c) if board[c][c] == 0 else self._first_empty(board)
        return best_move

    # ---------- Negamax with αβ + TT ----------
    def _negamax(self, board, depth, alpha, beta, to_move, last_move, root_depth):
        if time.time() > self.deadline:
            return (self._eval_fast(board, last_move), None)
        if depth == 0:
            return (self._eval_fast(board, last_move), None)

        key = self._hash(board, to_move)
        tt_hit = self.tt.get(key)
        if tt_hit and tt_hit[0] >= depth:
            # 用表中更深结果直接返回/作为排序提示
            _, s, bm = tt_hit
            return (s, bm)

        # 生成候选
        moves = self._candidates(board)
        if not moves:
            return (self._eval_fast(board, last_move), None)

        # 哈希走法优先（PV move），其次“快评”排序 + 历史启发
        pv = tt_hit[2] if tt_hit else None
        if pv in moves:
            moves.remove(pv)
            moves.insert(0, pv)

        moves = self._order(board, moves, to_move)

        best_score, best_move = -(10**9), None
        for r, c in moves:
            board[r][c] = to_move

            # 树内立即判胜 + 距离（赢快输慢）
            if self._is_win_local(board, r, c, to_move):
                ply = root_depth - depth
                score = (WIN - ply) if to_move == self.player else -(WIN - ply)
                board[r][c] = 0
                self.hist[(r, c)] = self.hist.get((r, c), 0) + 1
                self.tt[key] = (depth, score, (r, c))
                return (score, (r, c))

            s, _ = self._negamax(
                board, depth - 1, -beta, -alpha, -to_move, (r, c), root_depth
            )
            s = -s
            board[r][c] = 0

            if s > best_score:
                best_score, best_move = s, (r, c)
            if s > alpha:
                alpha = s
            if alpha >= beta:
                self.hist[(r, c)] = self.hist.get((r, c), 0) + 1  # 历史启发累加
                break

        self.tt[key] = (depth, best_score, best_move)
        return (best_score, best_move)

    # ---------- 候选生成与排序 ----------
    def _candidates(self, board, radius=2, topk=24):
        stones = [
            (i, j) for i in range(self.N) for j in range(self.N) if board[i][j] != 0
        ]
        if not stones:
            c = self.N // 2
            return [(c, c)]

        rmin = min(r for r, _ in stones)
        rmax = max(r for r, _ in stones)
        cmin = min(c for _, c in stones)
        cmax = max(c for _, c in stones)
        rmin, rmax = max(0, rmin - radius), min(self.N - 1, rmax + radius)
        cmin, cmax = max(0, cmin - radius), min(self.N - 1, cmax + radius)

        cand = []
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if board[r][c] == 0:
                    cand.append((r, c))

        # 简易战术优先：我方即胜、堵对手即胜
        wins = []
        blocks = []
        others = []
        for r, c in cand:
            board[r][c] = self.player
            if self._is_win_local(board, r, c, self.player):
                wins.append((r, c))
                board[r][c] = 0
                continue
            board[r][c] = 0
            board[r][c] = self.opponent
            if self._is_win_local(board, r, c, self.opponent):
                blocks.append((r, c))
                board[r][c] = 0
                continue
            board[r][c] = 0
            others.append((r, c))

        ordered = wins + blocks + others
        # 限制 topK
        return ordered[:topk]

    def _order(self, board, moves, to_move):
        # 快评（落子一次、只看该点四向字符串） + 历史启发 + 中心近邻
        scored = []
        for r, c in moves:
            board[r][c] = to_move
            s = self._eval_local_point(board, r, c, to_move)
            board[r][c] = 0
            h = self.hist.get((r, c), 0)
            cen = -((r - (self.N - 1) / 2) ** 2 + (c - (self.N - 1) / 2) ** 2)
            nb = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    rr, cc = r + dx, c + dy
                    if 0 <= rr < self.N and 0 <= cc < self.N and board[rr][cc] != 0:
                        nb += 1
            scored.append((s + h * 50, nb * 10 + cen, (r, c)))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [mv for _, __, mv in scored]

    # ---------- 评估（局部字符串，O(1)） ----------
    def _eval_fast(self, board, last_move):
        # 若无 last_move（根开始），给个轻微中心偏好
        if last_move is None:
            c = self.N // 2
            return -((c - (self.N - 1) / 2) ** 2)  # 很小的数，几乎不影响
        # Negamax 里我们始终以 self.player 为基准
        mr, mc = last_move
        # 只计算“最后一步”对双方的威胁强度（快而稳），取差值
        me = self._eval_local_point(board, mr, mc, self.player)
        opp = self._eval_local_point(board, mr, mc, self.opponent)
        return me - opp

    def _eval_local_point(self, board, r, c, who):
        # 以 (r,c) 为中心，四方向各取至多 4 格，生成字符串，然后匹配少量模式
        # who 的棋子 -> 'X'；对手 -> 'O'；空 -> '.'；边界 -> '#'
        def cell(rr, cc):
            if not (0 <= rr < self.N and 0 <= cc < self.N):
                return "#"
            v = board[rr][cc]
            if v == 0:
                return "."
            return "X" if v == who else "O"

        best = 0
        for dr, dc in DIRS:
            # 取中心前后各 4 格：共 9 字符
            line = []
            for k in range(-4, 5):
                line.append(cell(r + k * dr, c + k * dc))
            s = "".join(line)
            val = self._score_line_str(s)
            if val > best:
                best = val
        return best

    def _score_line_str(self, s):
        # 顺序很重要：由强到弱匹配一次即可
        if "XXXXX" in s:
            return PATTERN_SCORES["FIVE"]
        if ".XXXX." in s:
            return PATTERN_SCORES["LIVE_FOUR"]
        # 冲四（断点四）
        if ("XXX.X" in s) or ("XX.XX" in s) or (".XXXX" in s) or ("XXXX." in s):
            return PATTERN_SCORES["BROKEN_FOUR"]
        if ".XXX." in s:
            return PATTERN_SCORES["LIVE_THREE"]
        if (".XX.X." in s) or (".X.XX." in s):
            return PATTERN_SCORES["JUMP_THREE"]
        if (".XX.." in s) or ("..XX." in s) or (".X.X." in s):
            return PATTERN_SCORES["LIVE_TWO"]
        return 0

    # ---------- 胜负 / 即胜 ----------
    def _is_win_local(self, board, r, c, who):
        # 仅围绕 (r,c) 四方向统计连子
        for dr, dc in DIRS:
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.N and 0 <= cc < self.N and board[rr][cc] == who:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < self.N and 0 <= cc < self.N and board[rr][cc] == who:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= 5:
                return True
        return False

    def _find_immediate_win(self, board, who):
        for r in range(self.N):
            for c in range(self.N):
                if board[r][c] == 0:
                    board[r][c] = who
                    ok = self._is_win_local(board, r, c, who)
                    board[r][c] = 0
                    if ok:
                        return (r, c)
        return None

    # ---------- 其它 ----------
    def _first_empty(self, board):
        for i in range(self.N):
            for j in range(self.N):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _max_depth(self, board):
        total = self.N * self.N
        empties = sum(board[i][j] == 0 for i in range(self.N) for j in range(self.N))
        fill = (total - empties) / total
        if fill < 0.2:
            return 4
        if fill < 0.4:
            return 6
        if fill < 0.6:
            return 7
        return 8

    # ---------- Zobrist ----------
    def _init_zobrist(self, N):
        rnd = random.Random(2025)
        self.zkeys = [
            [[rnd.getrandbits(64) for _ in (0, 1)] for _ in range(N)] for __ in range(N)
        ]
        self.side_key = rnd.getrandbits(64)

    def _hash(self, board, to_move):
        h = 0
        for i in range(self.N):
            for j in range(self.N):
                v = board[i][j]
                if v == 0:
                    continue
                idx = 0 if v == 1 else 1
                h ^= self.zkeys[i][j][idx]
        if to_move == -1:
            h ^= self.side_key
        return h
