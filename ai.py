from agent import Agent
import numpy as np
import time


class Search(Agent):
    # ----------- 可调参数 -----------
    MAX_TIME = 58.5  # 每步最大用时（秒），略低于环境60秒的限制，留出安全余量
    START_DEPTH = 2  # 迭代加深起始深度
    MAX_DEPTH_CAP = 8  # 理论最大深度上限（会被时间打断）
    CAND_RADIUS = 2  # 候选点生成的邻域半径（切比雪夫距离）
    DEFENSE_WEIGHT = 1.10  # 敌方得分权重（>1 表示更重视防守）
    USE_LMR = True  # 是否启用晚着减深
    LMR_BASE_IDX = 3  # 前若干排序靠前的走法不做减深
    ENABLE_THREAT_EXTENSION = False  # 叶节点存在“四/活三”时的轻量扩展开关

    # 评估分值（可微调）
    SCORE_FIVE = 10_000_000
    SCORE_OPEN_FOUR = 200_000
    SCORE_CLOSED_FOUR = 50_000
    SCORE_OPEN_THREE = 4_000
    SCORE_SLEEP_THREE = 800
    SCORE_OPEN_TWO = 150

    # 置换表条目标记
    TT_EXACT = 0
    TT_LOWER = 1
    TT_UPPER = 2

    def __init__(self, player):
        super().__init__(player)
        self.size = None
        self.start_time = 0.0
        self.time_up = False
        self.zobrist = (
            None  # Zobrist 随机表: [2, size, size]，索引0表示玩家1，1表示玩家2
        )
        self.zobrist_turn = 0  # 行棋方随机量
        self.tt = {}  # 置换表: key -> (depth, flag, value, best_move)
        self.killers = {}  # killer moves: depth -> [(r,c), (r,c)]
        self.history = {}  # history heuristic: (player,r,c) -> score
        self.root_depth = 0
        self.best_move_last_completed = None

    # ================= 对外主入口 =================
    def make_move(self, board):
        # 初始化与时限
        self.size = len(board)
        self._ensure_zobrist()
        self.tt.clear()
        self.killers.clear()
        self.history.clear()
        self.time_up = False
        self.start_time = time.perf_counter()
        self.best_move_last_completed = None

        # 空棋盘：走中心
        stones = np.argwhere(board != 0)
        if stones.shape[0] == 0:
            c = self.size // 2
            return (c, c)

        # 立即胜 / 必堵：在搜索前做一次快速特判
        imm = self._immediate_move(board, self.player)
        if imm is not None:
            return imm

        # 迭代加深
        depth = self.START_DEPTH
        alpha, beta = -self.SCORE_FIVE, self.SCORE_FIVE
        last_best = None

        while depth <= self.MAX_DEPTH_CAP:
            if self._overtime():
                break

            self.root_depth = depth
            score, move = self._negamax(
                board, depth, alpha, beta, self.player, last_move=None, ply=0
            )

            if self.time_up:
                break

            if move is not None:
                self.best_move_last_completed = move
                last_best = move

            # 可选：吸气窗（若需要）
            # 这里简单处理：若击中边界，则扩大窗；否则精炼
            if score <= alpha:
                alpha = -self.SCORE_FIVE
            elif score >= beta:
                beta = self.SCORE_FIVE
            else:
                alpha = score - 2_000
                beta = score + 2_000

            depth += 1

        # 若因时限退出，回退到最近完成的解
        if self.best_move_last_completed is not None:
            return self.best_move_last_completed

        # 兜底：选择排序最高的候选
        cands = self._generate_candidates(board)
        return cands[0] if cands else None

    # ================= 搜索内核 =================
    def _negamax(self, board, depth, alpha, beta, player, last_move, ply):
        # 时间检查
        if self._overtime():
            self.time_up = True
            return 0, None

        opponent = 3 - player

        # 终局检测：上一步是否已经赢了
        if last_move is not None:
            r, c = last_move
            if self._is_five(board, r, c, opponent):
                # 对手已胜：对自身是巨大负分；ply 越小惩罚越重
                return -self.SCORE_FIVE + ply, None

        # 深度边界
        if depth == 0:
            val = self._evaluate(board)
            # 轻量威胁扩展（可选）
            if self.ENABLE_THREAT_EXTENSION and self._has_quick_threat(board):
                # 只扩展威胁着，避免爆炸
                return self._extend_threat(board, alpha, beta, player, last_move, ply)
            return val, None

        # 置换表查询
        key = self._hash(board, player)
        entry = self.tt.get(key)
        if entry is not None:
            et_depth, et_flag, et_val, et_move = entry
            if et_depth >= depth:
                if et_flag == self.TT_EXACT:
                    return et_val, et_move
                elif et_flag == self.TT_LOWER and et_val > alpha:
                    alpha = et_val
                elif et_flag == self.TT_UPPER and et_val < beta:
                    beta = et_val
                if alpha >= beta:
                    return et_val, et_move
            pv_move = entry[3]
        else:
            pv_move = None

        # 生成并排序走法
        moves = self._generate_candidates(board)
        if not moves:
            return 0, None

        # 将 PV/TT、killer、强应类走法提前
        moves = self._order_moves(board, moves, player, pv_move, depth)

        best_val = -self.SCORE_FIVE
        best_move = moves[0]
        original_alpha = alpha

        for idx, (r, c) in enumerate(moves):
            # 时间检查
            if self._overtime():
                self.time_up = True
                break

            # 着法落子
            board[r][c] = player

            # LMR：对排序较后的非强应走法尝试减深（保守）
            reduce = 0
            if (
                self.USE_LMR
                and depth >= 3
                and idx >= self.LMR_BASE_IDX
                and not self._is_forcing_move(board, r, c, player)
            ):
                reduce = 1

            if reduce > 0:
                val, _ = self._negamax(
                    board,
                    depth - 1 - reduce,
                    -beta,
                    -alpha,
                    3 - player,
                    (r, c),
                    ply + 1,
                )
                if val > alpha:
                    # 重新全深验证
                    val, _ = self._negamax(
                        board, depth - 1, -beta, -alpha, 3 - player, (r, c), ply + 1
                    )
            else:
                val, _ = self._negamax(
                    board, depth - 1, -beta, -alpha, 3 - player, (r, c), ply + 1
                )

            val = -val

            # 撤子
            board[r][c] = 0

            # 更新最优
            if val > best_val:
                best_val = val
                best_move = (r, c)

            if best_val > alpha:
                alpha = best_val

            # Beta 剪枝
            if alpha >= beta:
                # 记录 killer 与 history
                self._record_killer(depth, (r, c))
                self._record_history(player, r, c, depth)
                break

        # 写回置换表
        flag = self.TT_EXACT
        if best_val <= original_alpha:
            flag = self.TT_UPPER
        elif best_val >= beta:
            flag = self.TT_LOWER
        self.tt[key] = (depth, flag, best_val, best_move)

        return best_val, best_move

    # ================= 候选与排序 =================
    def _generate_candidates(self, board):
        # 空位太少直接返回所有空位（终局附近）
        empties = np.argwhere(board == 0)
        if empties.shape[0] <= 16:
            return [tuple(x) for x in empties.tolist()]

        # 邻域候选（半径 CAND_RADIUS）
        stones = np.argwhere(board != 0)
        if stones.shape[0] == 0:
            c = self.size // 2
            return [(c, c)]

        occ = set(map(tuple, stones.tolist()))
        cand = set()
        R = self.CAND_RADIUS
        for i, j in occ:
            r0 = max(0, i - R)
            r1 = min(self.size - 1, i + R)
            c0 = max(0, j - R)
            c1 = min(self.size - 1, j + R)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if board[r][c] == 0:
                        cand.add((r, c))

        # 排序：局部潜力 + 中央偏好
        def key_fn(pos):
            r, c = pos
            local_my = self._point_potential(board, r, c, self.player)
            local_op = self._point_potential(board, r, c, 3 - self.player)
            center_bias = -((r - self.size // 2) ** 2 + (c - self.size // 2) ** 2)
            return (max(local_my, local_op), local_my - 0.9 * local_op, center_bias)

        ordered = sorted(cand, key=key_fn, reverse=True)
        return ordered

    def _order_moves(self, board, moves, player, pv_move, depth):
        # 打表的 PV/TT 走先
        scores = {}
        for r, c in moves:
            board[r][c] = player
            if self._is_win_move(board, r, c, player):
                s = 10_000_000
            else:
                # 强应度：四/活三优先
                s = 0
                if self._creates_four(board, r, c, player):
                    s += 200_000
                if self._creates_open_three(board, r, c, player):
                    s += 20_000
                # 局部潜力
                s += self._point_potential(board, r, c, player)
                # 防守强应
                s += int(self._creates_four(board, r, c, 3 - player)) * 50_000
            board[r][c] = 0
            scores[(r, c)] = s

        killers = self.killers.get(depth, [])

        def cmp_key(pos):
            base = scores.get(pos, 0)
            pv = 1 if pv_move is not None and pos == pv_move else 0
            kil = 1 if pos in killers else 0
            return (pv, kil, base)

        return sorted(moves, key=cmp_key, reverse=True)

    # ================= 评估相关 =================
    def _evaluate(self, board):
        # 双边模式分 + 平滑项（最长连子）
        my = self._pattern_score(board, self.player)
        op = self._pattern_score(board, 3 - self.player)
        smooth = 0.02 * (
            self._max_chain(board, self.player)
            - self._max_chain(board, 3 - self.player)
        )
        return int(my - self.DEFENSE_WEIGHT * op + smooth * 100)

    def _pattern_score(self, board, who):
        # 将行、列、两条对角线转为字符串进行模式计数
        s = 0
        lines = []

        # 行
        lines.extend(board.tolist())
        # 列
        lines.extend(board.T.tolist())
        # 主对角线族
        for k in range(-self.size + 5, self.size - 4):  # 只抽长度≥5的对角线
            d = np.diag(board, k=k)
            if len(d) >= 5:
                lines.append(d.tolist())
        # 副对角线族
        flipped = np.fliplr(board)
        for k in range(-self.size + 5, self.size - 4):
            d = np.diag(flipped, k=k)
            if len(d) >= 5:
                lines.append(d.tolist())

        for arr in lines:
            line = self._line_to_str(arr, who)  # '0','1','2'：0空，1己，2敌
            s += self._score_line(line)

        return s

    def _line_to_str(self, arr, who):
        # who 的子记为 '1'，对手为 '2'，空为 '0'
        res = []
        opp = 3 - who
        for v in arr:
            if v == 0:
                res.append("0")
            elif v == who:
                res.append("1")
            else:
                res.append("2")
        return "".join(res)

    def _count_overlap(self, s, pat):
        # 允许重叠的计数
        cnt = 0
        start = 0
        while True:
            i = s.find(pat, start)
            if i == -1:
                break
            cnt += 1
            start = i + 1
        return cnt

    def _score_line(self, s):
        # 在字符串两端补 '2'，便于边界视作被堵
        s2 = f"2{s}2"
        score = 0

        # 五连
        score += self._count_overlap(s2, "11111") * self.SCORE_FIVE

        # 活四 / 冲四
        score += self._count_overlap(s2, "011110") * self.SCORE_OPEN_FOUR
        for pat in ["11110", "01111", "11011", "10111", "11101"]:
            score += self._count_overlap(s2, pat) * self.SCORE_CLOSED_FOUR

        # 活三 / 眠三（部分常见形）
        for pat in ["011100", "001110", "010110"]:
            score += self._count_overlap(s2, pat) * self.SCORE_OPEN_THREE
        for pat in [
            "001112",
            "211100",
            "011012",
            "210110",
            "010112",
            "211010",
            "0011102",
            "2011100",
        ]:
            score += self._count_overlap(s2, pat) * self.SCORE_SLEEP_THREE

        # 活二（取部分高频形）
        for pat in ["001100", "001010", "010010", "010100"]:
            score += self._count_overlap(s2, pat) * self.SCORE_OPEN_TWO

        return score

    def _max_chain(self, board, who):
        # 计算 who 的最长连续连子（四向）
        best = 0
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        pos = np.argwhere(board == who)
        for r, c in pos:
            for dr, dc in dirs:
                cnt = 1
                x, y = r + dr, c + dc
                while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                    cnt += 1
                    x += dr
                    y += dc
                x, y = r - dr, c - dc
                while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                    cnt += 1
                    x -= dr
                    y -= dc
                if cnt > best:
                    best = cnt
        return best

    # ================= 快速特判/局部函数 =================
    def _immediate_move(self, board, who):
        # 自己能直接赢：直接下
        cands = self._generate_candidates(board)
        for r, c in cands:
            board[r][c] = who
            if self._is_five(board, r, c, who):
                board[r][c] = 0
                return (r, c)
            board[r][c] = 0

        # 对手是否有立即胜？若有，先堵
        opp = 3 - who
        # 为控制成本，仅检测候选中的“对手立即胜”
        for r, c in cands:
            board[r][c] = opp
            if self._is_five(board, r, c, opp):
                board[r][c] = 0
                return (r, c)
            board[r][c] = 0
        return None

    def _is_five(self, board, r, c, who):
        # 检测 (r,c) 为 who 的落子后是否成五
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            cnt = 1
            x, y = r + dr, c + dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x += dr
                y += dc
            x, y = r - dr, c - dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x -= dr
                y -= dc
            if cnt >= 5:
                return True
        return False

    def _is_win_move(self, board, r, c, who):
        # 假设 board[r][c] 已经等于 who
        return self._is_five(board, r, c, who)

    def _creates_four(self, board, r, c, who):
        # 检测该点是否形成四（含冲四）
        res = False
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            cnt = 1
            open_ends = 0
            x, y = r + dr, c + dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x += dr
                y += dc
            if 0 <= x < self.size and 0 <= y < self.size and board[x][y] == 0:
                open_ends += 1
            x, y = r - dr, c - dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x -= dr
                y -= dc
            if 0 <= x < self.size and 0 <= y < self.size and board[x][y] == 0:
                open_ends += 1
            if cnt >= 4 and open_ends >= 1:
                res = True
                break
        return res

    def _creates_open_three(self, board, r, c, who):
        # 近似判定“活三”：连续3且两端均可延伸
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            cnt = 1
            open_ends = 0
            x, y = r + dr, c + dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x += dr
                y += dc
            if 0 <= x < self.size and 0 <= y < self.size and board[x][y] == 0:
                open_ends += 1
            x, y = r - dr, c - dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x -= dr
                y -= dc
            if 0 <= x < self.size and 0 <= y < self.size and board[x][y] == 0:
                open_ends += 1
            if cnt == 3 and open_ends == 2:
                return True
        return False

    def _point_potential(self, board, r, c, who):
        # 将 (r,c) 视作 who 落子，估计该点局部潜力（用于排序）
        score = 0
        opp = 3 - who
        board[r][c] = who
        if self._is_five(board, r, c, who):
            score += 1_000_000
        elif self._creates_four(board, r, c, who):
            score += 200_000
        elif self._creates_open_three(board, r, c, who):
            score += 20_000
        # 简单的最长连子近似（四向最大）
        score += 500 * self._local_max_chain(board, r, c, who)
        board[r][c] = 0

        # 防守：若对手在此能成强威胁，略降
        board[r][c] = opp
        if self._is_five(board, r, c, opp):
            score += 800_000  # 因为这是“必堵点”
        elif self._creates_four(board, r, c, opp):
            score += 50_000
        board[r][c] = 0

        return score

    def _local_max_chain(self, board, r, c, who):
        best = 1
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            cnt = 1
            x, y = r + dr, c + dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x += dr
                y += dc
            x, y = r - dr, c - dc
            while 0 <= x < self.size and 0 <= y < self.size and board[x][y] == who:
                cnt += 1
                x -= dr
                y -= dc
            if cnt > best:
                best = cnt
        return best

    def _is_forcing_move(self, board, r, c, who):
        # 下完后是否构成强应（四/活三/必堵）
        if self._is_win_move(board, r, c, who):
            return True
        if self._creates_four(board, r, c, who):
            return True
        if self._creates_open_three(board, r, c, who):
            return True
        # 对对手的必堵点
        opp = 3 - who
        board[r][c] = opp
        v = self._is_five(board, r, c, opp) or self._creates_four(board, r, c, opp)
        board[r][c] = 0
        return v

    def _has_quick_threat(self, board):
        # 粗略检测全局是否存在“四/活三”
        lines = []

        lines.extend(board.tolist())
        lines.extend(board.T.tolist())
        for k in range(-self.size + 5, self.size - 4):
            d = np.diag(board, k=k)
            if len(d) >= 5:
                lines.append(d.tolist())
        flipped = np.fliplr(board)
        for k in range(-self.size + 5, self.size - 4):
            d = np.diag(flipped, k=k)
            if len(d) >= 5:
                lines.append(d.tolist())

        def has_pat(line):
            s = "".join(str(x) for x in line)
            return ("1111" in s) or ("2222" in s)

        return any(has_pat(a) for a in lines)

    def _extend_threat(self, board, alpha, beta, player, last_move, ply):
        # 轻量威胁扩展：仅扩展能形成四/活三的走法
        moves = self._generate_candidates(board)
        moves = [m for m in moves if self._would_be_threat(board, m, player)]
        if not moves:
            return self._evaluate(board), None

        best = -self.SCORE_FIVE
        for r, c in moves:
            if self._overtime():
                break
            board[r][c] = player
            val, _ = self._negamax(board, 1, -beta, -alpha, 3 - player, (r, c), ply + 1)
            val = -val
            board[r][c] = 0
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best, None

    def _would_be_threat(self, board, move, who):
        r, c = move
        board[r][c] = who
        v = self._creates_four(board, r, c, who) or self._creates_open_three(
            board, r, c, who
        )
        board[r][c] = 0
        return v

    # ================= 置换表 / Zobrist =================
    def _ensure_zobrist(self):
        if self.zobrist is not None and self.zobrist.shape[1] == self.size:
            return
        # 生成稳定的 64-bit 随机表
        rng = np.random.default_rng(2025)  # 固定种子，便于复现实验
        # 注意：部分 numpy 版本在 integers 的边界处理上对 uint64 支持不完整，
        # 这里使用 int64 安全上界再转换为 uint64，避免 "high is out of bounds for int64" 异常
        max_int64 = np.iinfo(np.int64).max
        self.zobrist = rng.integers(
            low=1,
            high=max_int64,
            size=(2, self.size, self.size),
            dtype=np.int64,
        ).astype(np.uint64)
        self.zobrist_turn = np.uint64(
            rng.integers(low=1, high=max_int64, size=(), dtype=np.int64)
        )

    def _hash(self, board, player_to_move):
        h = np.uint64(0)
        # 玩家1 -> idx 0，玩家2 -> idx 1
        p1 = np.argwhere(board == 1)
        for r, c in p1:
            h ^= self.zobrist[0, r, c]
        p2 = np.argwhere(board == 2)
        for r, c in p2:
            h ^= self.zobrist[1, r, c]
        if player_to_move == 1:
            h ^= self.zobrist_turn
        return int(h)

    def _record_killer(self, depth, move):
        lst = self.killers.get(depth, [])
        if move in lst:
            return
        lst = [move] + lst
        if len(lst) > 2:
            lst = lst[:2]
        self.killers[depth] = lst

    def _record_history(self, player, r, c, depth):
        key = (player, r, c)
        self.history[key] = self.history.get(key, 0) + depth * depth

    # ================= 时间控制 =================
    def _overtime(self):
        return (time.perf_counter() - self.start_time) >= self.MAX_TIME
