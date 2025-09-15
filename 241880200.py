from agent import Agent
import numpy as np
import time


"""主要思路: 以minimax搜索、alpha-beta剪枝为基础, 逐步完善动态深度限制和启发式评估函数, 使用滑窗卷积统计己方/对方的棋形并加权。"""


"""评分表语义
- LIVE_FOUR: 连四且两端皆空（双活口）
- DEAD_FOUR: 连四且仅一端为空（单活口、半开放四）
- BROKEN_FOUR: 断点四（1-1-0-1-1 型），至少一端可延伸即为强威胁
- SLEEPY_THREE: 眠三/跳三（含开放跳三），潜在形成冲四
注：组合加分基于计数的威胁叠加，而非单个窗口评分。
"""
SCORE_TABLE = {
    "WIN": 1000000,  # 五连
    "LIVE_FOUR": 50000,  # 活四（双活口）
    "BROKEN_FOUR": 45000,  # 断点四/冲四：一步成五或逼应
    "DOUBLE_DEAD_FOUR": 50000,  # 任意两处四类威胁并存（含断点四/眠四）
    "DEAD_FOUR_LIVE_THREE": 50000,  # 四类威胁+活三并存
    "DOUBLE_LIVE_THREE": 10000,  # 双活三
    "LIVE_THREE": 8000,  # 活三（双活口）
    "SLEEPY_THREE": 1000,  # 眠三/跳三
    "LIVE_TWO": 100,  # 活二
    "DEAD_FOUR": 50,  # 眠四（单活口）
    "DEAD_THREE": 10,  # 死三
    "DEAD_TWO": 2,  # 死二
}

# 棋形表示例（参考用，不在评估中直接使用）
KERNELS = {
    # 五连
    "WIN": [np.array([1, 1, 1, 1, 1], dtype=np.int8)],
    # 活四: 两端皆空
    "LIVE_FOUR": [np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)],
    # 眠四(单边被堵)
    "DEAD_FOUR": [
        np.array([-1, 1, 1, 1, 1, 0], dtype=np.int8),
        np.array([0, 1, 1, 1, 1, -1], dtype=np.int8),
    ],
    # 活三: 两端皆空
    "LIVE_THREE": [np.array([0, 1, 1, 1, 0], dtype=np.int8)],
    # 跳三(破三): 两端通常为空, 内部有断点
    "BROKEN_THREE": [
        np.array([1, 1, 0, 1, 0], dtype=np.int8),
        np.array([1, 0, 1, 1, 0], dtype=np.int8),
        np.array([0, 1, 1, 0, 1], dtype=np.int8),
        np.array([0, 1, 0, 1, 1], dtype=np.int8),
    ],
    # 开放跳三(6格窗口): 两端为空, 中间三子带空格
    "BROKEN_THREE_OPEN": [
        np.array([0, 1, 1, 0, 1, 0], dtype=np.int8),
        np.array([0, 1, 0, 1, 1, 0], dtype=np.int8),
    ],
    # 眠三: 单边被堵
    "DEAD_THREE": [
        np.array([-1, 1, 1, 1, 0], dtype=np.int8),
        np.array([0, 1, 1, 1, -1], dtype=np.int8),
    ],
    # 活二: 两端皆空
    "LIVE_TWO": [
        np.array([0, 1, 1, 0, 0], dtype=np.int8),
        np.array([0, 0, 1, 1, 0], dtype=np.int8),
        np.array([0, 1, 0, 1, 0], dtype=np.int8),
    ],
    # 眠二: 单边被堵
    "DEAD_TWO": [
        np.array([-1, 1, 1, 0, 0], dtype=np.int8),
        np.array([0, 0, 1, 1, -1], dtype=np.int8),
        np.array([-1, 1, 0, 1, 0], dtype=np.int8),
        np.array([0, 1, 0, 1, -1], dtype=np.int8),
    ],
}


class Search(Agent):
    def __init__(self, player):
        super().__init__(player)

    def make_move(self, board):
        """使用迭代加深 + alpha-beta + 候选生成, 在时间限制内返回一步。"""
        # 留出安全余量, 防止超时 (外部限制为60s)
        time_budget_sec = 58.5
        best_move = self.iterative_deepening(board, time_budget_sec)
        if best_move is None:
            moves = self.get_moves(board)
            return moves[0] if moves else None
        return best_move

    def max_depth(self, board):
        """动态设置迭代加深的最大上限(软上限), 实际深度由时间裁剪。
        依据棋盘空位比例调节目标深度, 以便在中后盘尝试更深。
        """
        n = len(board)
        total = n * n
        empties = int(np.count_nonzero(board == 0))
        fill_ratio = (total - empties) / total
        if fill_ratio < 0.2:
            return 3
        elif fill_ratio < 0.4:
            return 4
        elif fill_ratio < 0.6:
            return 5
        return 6

    def iterative_deepening(self, board, time_limit_s: float):
        """迭代加深搜索, 在给定时间内逐层加深, 返回已完成层的最佳着法。
        - 若在某层用尽时间, 返回上一层的结果。
        - 使用 self._deadline 供递归检查超时。
        """
        self._deadline = time.time() + time_limit_s

        best_move = None
        best_score = None
        depth_cap = self.max_depth(board)

        for depth in range(1, depth_cap + 1):
            # 每一层开始时简单检查剩余时间
            if time.time() > self._deadline:
                break
            score, move = self.minimax(board, depth, float("-inf"), float("inf"), True)
            # 若该层因为超时导致未给出move, 直接退出并返回上一层结果
            if move is None:
                break
            best_move, best_score = move, score

        # 清理deadline标记
        self._deadline = None
        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """带 alpha-beta 剪枝的极小极大搜索, 返回 (score, move)
        - is_maximizing=True 表示当前轮到 self.player
        - 使用 alpha/beta 进行剪枝, 并进行启发式走法排序
        - 以 self.evaluate 的相对分数作为评价
        """
        # 超时快速返回: 返回当前评估与 None 表示未完成该分支
        if hasattr(self, "_deadline") and self._deadline is not None:
            if time.time() > self._deadline:
                return (self.evaluate(board), None)
        # 终止条件: 深度用尽
        if depth == 0:
            return (self.evaluate(board), None)

        # 候选步生成: 仅在已有棋子附近选点, 降低分支因子
        legal_moves = self.get_candidate_moves(board)
        if not legal_moves:
            return (self.evaluate(board), None)

        # 走法排序: 先考虑更优的候选步, 提高剪枝效率
        ordered_moves = self.order_moves(board, legal_moves, is_maximizing)

        if is_maximizing:
            best_score = float("-inf")
            best_move = None
            for r, c in ordered_moves:
                # 模拟我方落子
                board[r][c] = self.player
                child_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board[r][c] = 0  # 回溯
                if child_score > best_score:
                    best_score = child_score
                    best_move = (r, c)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # beta剪枝
            return (best_score, best_move)
        else:
            best_score = float("inf")
            best_move = None
            for r, c in ordered_moves:
                # 模拟对手落子
                board[r][c] = self.opponent
                child_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board[r][c] = 0
                if child_score < best_score:
                    best_score = child_score
                    best_move = (r, c)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # alpha剪枝
            return (best_score, best_move)

    def get_moves(self, board):
        """生成所有空位。"""
        return [
            (i, j)
            for i in range(len(board))
            for j in range(len(board))
            if board[i][j] == 0
        ]

    def get_candidate_moves(self, board, radius=2):
        """候选步生成: 只考虑距离任一已有棋子不超过 radius 的空位。"""
        n = len(board)
        stones = np.argwhere(board != 0)
        if stones.size == 0:
            center = n // 2
            return [(center, center)]
        candidates = set()
        for x, y in stones:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and board[nx][ny] == 0:
                        candidates.add((nx, ny))
        return list(candidates)

    def order_moves(self, board, moves, is_maximizing):
        """走法排序: 轻量启发式, 优先靠近中心且周围子多的点。"""
        n = len(board)
        center = (n - 1) / 2.0

        def neighbor_count(r, c):
            # 统计8邻域内非空子数量
            count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < n and 0 <= nc < n and board[nr][nc] != 0:
                        count += 1
            return count

        def center_score(r, c):
            # 越靠近中心分数越高
            return -((r - center) ** 2 + (c - center) ** 2)

        scored = []
        for r, c in moves:
            local = neighbor_count(r, c)
            cent = center_score(r, c)
            # 组合分数: 邻居权重大一些
            scored.append(((local * 100) + cent, (r, c)))

        # 高分优先
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]

    # 移除 get_tactical_moves 及相关辅助，改由评估分系统完全驱动

    def evaluate(self, board):
        """卷积式评估: 统一使用长度6的滑窗, 并对每条线做边界填充 2 后遍历行/列/对角线。
        统计己方/对方的活四、眠四、活三、跳三、眠三、活二、眠二等棋形并加权; 若出现五连, 立即返回极值分。
        """
        n = len(board)

        def lines():
            # 行
            for i in range(n):
                yield board[i, :]
            # 列
            for j in range(n):
                yield board[:, j]
            # 正对角线(斜下): 全部对角线
            for k in range(-(n - 1), n):
                yield np.diag(board, k=k)
            # 反对角线(斜上): 全部对角线
            flipped = np.fliplr(board)
            for k in range(-(n - 1), n):
                yield np.diag(flipped, k=k)

        def score_window(mapped):
            """对6格窗口(相对视角: 1/0/-1)评分, 并统计组合棋形计数。
            返回: (score, feats_me, feats_opp)
            """
            # 五连
            if np.all(mapped[:5] == 1) or np.all(mapped[1:] == 1):
                return (
                    SCORE_TABLE["WIN"],
                    {
                        "LIVE_FOUR": 0,
                        "DEAD_FOUR": 0,
                        "LIVE_THREE": 0,
                        "SLEEPY_THREE": 0,
                    },
                    {
                        "LIVE_FOUR": 0,
                        "DEAD_FOUR": 0,
                        "LIVE_THREE": 0,
                        "SLEEPY_THREE": 0,
                    },
                )
            if np.all(mapped[:5] == -1) or np.all(mapped[1:] == -1):
                return (
                    -SCORE_TABLE["WIN"],
                    {
                        "LIVE_FOUR": 0,
                        "DEAD_FOUR": 0,
                        "LIVE_THREE": 0,
                        "SLEEPY_THREE": 0,
                    },
                    {
                        "LIVE_FOUR": 0,
                        "DEAD_FOUR": 0,
                        "LIVE_THREE": 0,
                        "SLEEPY_THREE": 0,
                    },
                )

            def scan_runs(target):
                score = 0
                feats = {
                    "LIVE_FOUR": 0,
                    "DEAD_FOUR": 0,
                    "BROKEN_FOUR": 0,
                    "LIVE_THREE": 0,
                    "SLEEPY_THREE": 0,
                }
                i = 0
                while i < 6:
                    if mapped[i] == target:
                        j = i
                        while j < 6 and mapped[j] == target:
                            j += 1
                        length = j - i
                        left_open = (i - 1 >= 0) and (mapped[i - 1] == 0)
                        right_open = (j < 6) and (mapped[j] == 0)
                        open_ends = int(left_open) + int(right_open)
                        if length == 4:
                            if open_ends == 2:
                                score += SCORE_TABLE["LIVE_FOUR"]
                                feats["LIVE_FOUR"] += 1
                            elif open_ends == 1:
                                score += SCORE_TABLE["DEAD_FOUR"]
                                feats["DEAD_FOUR"] += 1
                        elif length == 3:
                            if open_ends == 2:
                                score += SCORE_TABLE["LIVE_THREE"]
                                feats["LIVE_THREE"] += 1
                            elif open_ends == 1:
                                score += SCORE_TABLE["SLEEPY_THREE"]
                                feats["SLEEPY_THREE"] += 1
                            else:  # open_ends == 0
                                score += SCORE_TABLE["DEAD_THREE"]
                        elif length == 2:
                            if open_ends == 2:
                                score += SCORE_TABLE["LIVE_TWO"]
                            elif open_ends == 1:
                                score += SCORE_TABLE["DEAD_TWO"]
                        i = j
                    else:
                        i += 1

                # 跳三/冲四: 形如 段-空-段, 总子数=3/4
                for g in range(1, 5):
                    if mapped[g] != 0:
                        continue
                    # 左段
                    l = 0
                    k = g - 1
                    while k >= 0 and mapped[k] == target:
                        l += 1
                        k -= 1
                    # 右段
                    r = 0
                    k = g + 1
                    while k < 6 and mapped[k] == target:
                        r += 1
                        k += 1
                    total_len = l + r
                    if total_len == 3:
                        left_open = (g - l - 1 >= 0) and (mapped[g - l - 1] == 0)
                        right_open = (g + r + 1 < 6) and (mapped[g + r + 1] == 0)
                        open_ends = int(left_open) + int(right_open)
                        if open_ends == 2:
                            # 开放跳三（两端皆空）力度可等同活三或眠三, 这里记为眠三权重
                            score += SCORE_TABLE["SLEEPY_THREE"]
                            feats["SLEEPY_THREE"] += 1
                        elif open_ends == 1:
                            score += SCORE_TABLE["SLEEPY_THREE"]
                            feats["SLEEPY_THREE"] += 1
                        else:  # open_ends == 0
                            score += SCORE_TABLE["DEAD_THREE"]
                    elif total_len == 4:
                        # 冲四/断点四: 1-1-0-1-1，至少一端可延伸即为强威胁
                        left_open = (g - l - 1 >= 0) and (mapped[g - l - 1] == 0)
                        right_open = (g + r + 1 < 6) and (mapped[g + r + 1] == 0)
                        if left_open or right_open:
                            score += SCORE_TABLE["BROKEN_FOUR"]
                            feats["BROKEN_FOUR"] += 1
                return score, feats

            me_score, me_feats = scan_runs(1)
            opp_score, opp_feats = scan_runs(-1)
            return (me_score - opp_score, me_feats, opp_feats)

        total = 0
        me = self.player
        # 统计用于组合棋形的计数（我方与对方各自）
        combo_me = {
            "LIVE_FOUR": 0,
            "DEAD_FOUR": 0,
            "BROKEN_FOUR": 0,
            "LIVE_THREE": 0,
            "SLEEPY_THREE": 0,
        }
        combo_opp = {
            "LIVE_FOUR": 0,
            "DEAD_FOUR": 0,
            "BROKEN_FOUR": 0,
            "LIVE_THREE": 0,
            "SLEEPY_THREE": 0,
        }

        # 行/列/对角线上统一滑窗: 全部转换为相对视角并做边界填充(中性哨兵), 仅用6格窗口
        for line in lines():
            raw = np.array(line, dtype=int)
            # 相对映射: 我方=1, 对方=-1, 空=0
            mapped_line = np.where(raw == me, 1, np.where(raw == 0, 0, -1)).astype(
                np.int8
            )
            # 边界填充: 使用中性哨兵(既非我方也非对方, 也非空), 避免被误认为对手
            BORDER_SENTINEL = 2
            padded = np.pad(mapped_line, (1, 1), constant_values=BORDER_SENTINEL)
            for i in range(0, len(padded) - 6 + 1):
                win = padded[i : i + 6]
                s, feats_me, feats_opp = score_window(win)
                if s >= SCORE_TABLE["WIN"] or s <= -SCORE_TABLE["WIN"]:
                    return s
                total += s
                # 组合计数累积
                for k in feats_me:
                    combo_me[k] += feats_me[k]
                    combo_opp[k] += feats_opp[k]

        # 组合型
        def combo_bonus(cnt):
            bonus = 0
            four_threats = cnt["LIVE_FOUR"] + cnt["DEAD_FOUR"] + cnt["BROKEN_FOUR"]
            if four_threats >= 2 and "DOUBLE_DEAD_FOUR" in SCORE_TABLE:
                bonus += SCORE_TABLE["DOUBLE_DEAD_FOUR"]
            if (
                four_threats >= 1
                and cnt["LIVE_THREE"] >= 1
                and "DEAD_FOUR_LIVE_THREE" in SCORE_TABLE
            ):
                bonus += SCORE_TABLE["DEAD_FOUR_LIVE_THREE"]
            if cnt["LIVE_THREE"] >= 2 and "DOUBLE_LIVE_THREE" in SCORE_TABLE:
                bonus += SCORE_TABLE["DOUBLE_LIVE_THREE"]
            return bonus

        total += combo_bonus(combo_me)
        total -= combo_bonus(combo_opp)

        return int(total)
