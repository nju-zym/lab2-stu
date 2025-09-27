"""Gomoku 搜索引擎基础框架（单文件版本）。

在保持接口简洁的前提下搭建可运行的搜索 AI 骨架，核心特性：
- 迭代加深 + Negamax + Alpha-Beta 剪枝
- 启发式走法排序
- 基于行扫描的静态评估
- 为置换表、Zobrist 哈希、增量评估等高级优化预留 TODO

当前实现旨在先得到稳定可用的 AI，后续可在指定位置扩展性能优化。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import time

import numpy as np

from agent import Agent


class Search(Agent):
    """单文件的五子棋搜索 AI 框架。"""

    # ------------------------------
    # 对外接口区域
    # ------------------------------

    def __init__(
        self,
        player: int,
        board_size: int = 15,
        time_limit: float = 55.0,
    ) -> None:
        """初始化搜索器。"""
        super().__init__(player)
        self.board_size = board_size
        self.time_limit = time_limit

        # 评分常量，可按需调整。
        self._win_score = 1_000_000
        self._pattern_weights = {
            (5, 0): self._win_score,
            (4, 2): 10_000,
            (4, 1): 3_000,
            (3, 2): 800,
            (3, 1): 150,
            (2, 2): 50,
            (2, 1): 10,
        }

        # 搜索深度上限（TODO: 根据时间和局面动态调整）。
        self._max_depth_upper = 4

        # 预留的数据结构
        self._zobrist_table: Optional[np.ndarray] = None  # Zobrist 哈希表
        self._transposition_table: Dict[int, Dict[str, object]] = {}  # 置换表缓存
        self._pattern_cache: Dict[str, np.ndarray] = {}  # 棋形卷积核缓存
        self._history_heuristic: Dict[Tuple[int, int], float] = {}  # 历史启发表

    def make_move(self, board: np.ndarray) -> Optional[Tuple[int, int]]:
        """代理接口：接收棋盘矩阵，返回落子坐标。"""
        board_state = self._build_state(board)
        deadline = time.time() + self.time_limit
        best_move, _ = self._iterative_deepening(board_state, deadline)

        if best_move is None:
            fallback_moves = list(self._generate_moves(board_state))
            best_move = fallback_moves[0] if fallback_moves else None

        return best_move

    # ------------------------------
    # 核心搜索逻辑
    # ------------------------------

    def _iterative_deepening(
        self,
        board_state: Dict[str, object],
        deadline: float,
    ) -> Tuple[Optional[Tuple[int, int]], Dict[str, object]]:
        """迭代加深主循环，按深度逐层扩展。"""

        best_move: Optional[Tuple[int, int]] = None
        best_score = float("-inf")
        principal_variation: List[Tuple[int, int]] = []
        depth_limit = self._select_search_depth(board_state)

        info: Dict[str, object] = {
            "score": 0.0,
            "depth": 0,
            "principal_variation": principal_variation,
        }

        for depth in range(1, depth_limit + 1):
            if self._check_timeout(deadline):
                break

            score, move, pv_line = self._negamax(
                depth,
                -float("inf"),
                float("inf"),
                board_state,
                deadline,
                principal_variation,
            )

            if self._check_timeout(deadline):
                break

            info.update(
                {
                    "score": score,
                    "depth": depth,
                    "principal_variation": [move, *pv_line] if move else pv_line,
                }
            )

            if move is not None:
                best_move = move
                best_score = score
                principal_variation = [move, *pv_line]

            if best_score >= self._win_score:
                break

        return best_move, info

    def _negamax(
        self,
        depth: int,
        alpha: float,
        beta: float,
        board_state: Dict[str, object],
        deadline: float,
        principal_variation: Sequence[Tuple[int, int]] = (),
    ) -> Tuple[float, Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """Negamax + Alpha-Beta 剪枝的递归核心。"""

        if self._check_timeout(deadline):
            return self._evaluate_board(board_state), None, []

        if depth == 0 or self._is_terminal(board_state):
            return self._evaluate_board(board_state), None, []

        best_score = float("-inf")
        best_move: Optional[Tuple[int, int]] = None
        best_line: List[Tuple[int, int]] = []

        moves = list(self._generate_moves(board_state))
        if not moves:
            return self._evaluate_board(board_state), None, []

        ordered_moves = self._order_moves(moves, board_state, principal_variation)

        for move in ordered_moves:
            self._apply_move(board_state, move)

            if principal_variation and principal_variation[0] == move:
                child_pv = principal_variation[1:]
            else:
                child_pv = ()

            score, _, child_line = self._negamax(
                depth - 1,
                -beta,
                -alpha,
                board_state,
                deadline,
                child_pv,
            )

            score = -score

            self._undo_move(board_state)

            if score > best_score:
                best_score = score
                best_move = move
                best_line = [move, *child_line]

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

            if self._check_timeout(deadline):
                break

        return best_score, best_move, best_line[1:] if best_line else []

    # ------------------------------
    # 走法生成与排序
    # ------------------------------

    def _generate_moves(
        self,
        board_state: Dict[str, object],
    ) -> Iterable[Tuple[int, int]]:
        """生成当前局面的候选走法列表。"""

        board = board_state["board"]
        size = board_state["size"]

        if board_state["empty_count"] == 0:
            return []

        if board_state["empty_count"] == size * size:
            center = (size // 2, size // 2)
            return [center]

        occupied = np.argwhere(board != 0)
        candidates: Set[Tuple[int, int]] = set()
        radius = 2

        for r, c in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                        candidates.add((nr, nc))

        if not candidates:
            empties = np.argwhere(board == 0)
            return [(int(r), int(c)) for r, c in empties]

        return list(candidates)

    def _order_moves(
        self,
        moves: Iterable[Tuple[int, int]],
        board_state: Dict[str, object],
        principal_variation: Sequence[Tuple[int, int]] = (),
    ) -> List[Tuple[int, int]]:
        """结合多种启发式对走法按优先级排序。"""

        board = board_state["board"]
        player = board_state["current_player"]
        opponent = self._opponent(player)
        size = board_state["size"]
        center = (size - 1) / 2.0

        pv_set: Set[Tuple[int, int]] = {mv for mv in principal_variation}
        scored_moves: List[Tuple[float, Tuple[int, int]]] = []

        for move in moves:
            row, col = move
            bonus = 0.0

            if move in pv_set:
                bonus += 10_000

            center_distance = abs(row - center) + abs(col - center)
            score = -center_distance

            if board[row, col] == 0:
                board[row, col] = player
                if self._check_win(board, row, col, player):
                    board[row, col] = 0
                    scored_moves.append((self._win_score + bonus, move))
                    continue
                board[row, col] = 0

                board[row, col] = opponent
                if self._check_win(board, row, col, opponent):
                    bonus += 5_000
                board[row, col] = 0

            score += bonus
            score += self._estimate_move_potential(board, move, player)
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves]

    # ------------------------------
    # 评估函数相关
    # ------------------------------

    def _evaluate_board(
        self,
        board_state: Dict[str, object],
    ) -> float:
        """静态评估：返回当前局面的数值评分。"""

        board = board_state["board"]
        player = board_state["current_player"]
        opponent = self._opponent(player)

        if self._has_five(board, player):
            return self._win_score
        if self._has_five(board, opponent):
            return -self._win_score

        player_score = self._score_player(board, player)
        opponent_score = self._score_player(board, opponent)

        return player_score - opponent_score

    def _incremental_update(
        self,
        board_state: Dict[str, object],
        move: Tuple[int, int],
        previous_score: float,
    ) -> float:
        """增量评估：基于上一分数快速更新。"""
        # TODO: 实现局部增量更新。当前回退为全量评估。
        return self._evaluate_board(board_state)

    def _detect_patterns(
        self,
        board_array: np.ndarray,
    ) -> Dict[str, int]:
        """统计棋盘上的关键棋形，用于评分。"""
        # TODO: 引入更丰富的棋形检测。
        return {}

    # ------------------------------
    # 局面变换与回溯
    # ------------------------------

    def _apply_move(
        self,
        board_state: Dict[str, object],
        move: Tuple[int, int],
    ) -> None:
        """在当前局面上执行走法，并更新辅助信息。"""

        board = board_state["board"]
        row, col = move
        player = board_state["current_player"]

        if board[row, col] != 0:
            raise ValueError(f"非法落子位置: {move}")

        record = {
            "move": move,
            "player": player,
            "previous_last_move": board_state.get("last_move"),
            "previous_last_player": board_state.get("last_player"),
        }

        board_state["move_stack"].append(record)

        board[row, col] = player
        board_state["last_move"] = move
        board_state["last_player"] = player
        board_state["current_player"] = self._opponent(player)
        board_state["empty_count"] -= 1

        # TODO: 在此接入Zobrist哈希及置换表更新。

    def _undo_move(
        self,
        board_state: Dict[str, object],
    ) -> None:
        """撤销最近的走法，恢复到之前的状态。"""

        if not board_state["move_stack"]:
            return

        record = board_state["move_stack"].pop()
        row, col = record["move"]
        board = board_state["board"]

        board[row, col] = 0
        board_state["current_player"] = record["player"]
        board_state["last_move"] = record["previous_last_move"]
        board_state["last_player"] = record["previous_last_player"]
        board_state["empty_count"] += 1

        # TODO: 回滚Zobrist哈希及置换表状态。

    # ------------------------------
    # 置换表与哈希
    # ------------------------------

    def _probe_transposition(
        self,
        board_state: Dict[str, object],
        depth: int,
        alpha: float,
        beta: float,
    ) -> Optional[Tuple[float, Optional[Tuple[int, int]], float, float]]:
        """置换表查询：返回缓存的分数与搜索窗口。"""
        # TODO: 接入真实的置换表查询逻辑。
        return None

    def _store_transposition(
        self,
        board_state: Dict[str, object],
        depth: int,
        score: float,
        flag: str,
        best_move: Optional[Tuple[int, int]],
    ) -> None:
        """置换表写入：缓存已搜索的局面信息。"""
        # TODO: 接入真实的置换表写入逻辑。
        return

    def _ensure_zobrist(self, board_size: int) -> None:
        """初始化或重置 Zobrist 哈希表。"""
        # TODO: 生成随机位串用于哈希，并配合置换表使用。
        if self._zobrist_table is None or self._zobrist_table.shape[0] != board_size:
            # TODO: 使用高质量随机表。目前以简单占位数据替代，避免初始化开销。
            self._zobrist_table = np.zeros((board_size, board_size, 3), dtype=np.uint64)

    def _update_hash(
        self,
        board_state: Dict[str, object],
        move: Tuple[int, int],
        player: int,
    ) -> int:
        """增量更新当前哈希值。"""
        # TODO: 实现基于Zobrist的哈希增量更新。
        return 0

    # ------------------------------
    # 时间控制与终止条件
    # ------------------------------

    def _check_timeout(self, deadline: float) -> bool:
        """判断是否触发超时，供搜索在递归中及时返回。"""
        return time.time() >= deadline

    def _is_terminal(
        self,
        board_state: Dict[str, object],
    ) -> bool:
        """检查当前局面是否已分出胜负或棋满。"""

        board = board_state["board"]

        if self._has_five(board, 1) or self._has_five(board, 2):
            return True

        return board_state["empty_count"] == 0

    # ------------------------------
    # 辅助构造与工具函数
    # ------------------------------

    def _build_state(self, board: np.ndarray) -> Dict[str, object]:
        """将外部棋盘矩阵包装为内部状态字典。"""

        board_array = np.asarray(board, dtype=int)
        if board_array.ndim != 2 or board_array.shape[0] != board_array.shape[1]:
            raise ValueError("棋盘必须为方形二维数组")

        size = board_array.shape[0]

        self.board_size = size
        self._ensure_zobrist(size)

        stones_player1 = int(np.count_nonzero(board_array == 1))
        stones_player2 = int(np.count_nonzero(board_array == 2))

        if not (
            stones_player1 == stones_player2 or stones_player1 == stones_player2 + 1
        ):
            raise ValueError("棋盘状态异常：棋子数量差异不合法")

        current_player = 1 if stones_player1 == stones_player2 else 2
        empty_count = size * size - stones_player1 - stones_player2

        state: Dict[str, object] = {
            "board": board_array.copy(),
            "size": size,
            "current_player": current_player,
            "last_move": None,
            "last_player": None,
            "move_stack": [],
            "empty_count": empty_count,
        }

        return state

    def _opponent(self, player: int) -> int:
        """获取对手编号。"""
        return 1 if player == 2 else 2

    # ------------------------------
    # 内部工具函数
    # ------------------------------

    def _select_search_depth(self, board_state: Dict[str, object]) -> int:
        """根据局面稠密度和时间预算选择搜索深度。"""

        empties = board_state["empty_count"]
        total = board_state["size"] ** 2

        if empties > total * 0.6:
            return 2
        if empties > total * 0.3:
            return 4
        return min(self._max_depth_upper, 4)

    def _estimate_move_potential(
        self,
        board: np.ndarray,
        move: Tuple[int, int],
        player: int,
    ) -> float:
        """基于四个方向的连续数估算走法价值。"""

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        r, c = move
        potential = 0.0

        for dr, dc in directions:
            count = 1

            nr, nc = r + dr, c + dc
            while (
                0 <= nr < board.shape[0]
                and 0 <= nc < board.shape[1]
                and board[nr, nc] == player
            ):
                count += 1
                nr += dr
                nc += dc

            nr, nc = r - dr, c - dc
            while (
                0 <= nr < board.shape[0]
                and 0 <= nc < board.shape[1]
                and board[nr, nc] == player
            ):
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                potential += self._win_score
            elif count == 4:
                potential += 500
            elif count == 3:
                potential += 80
            elif count == 2:
                potential += 15

        return potential

    def _score_player(self, board: np.ndarray, player: int) -> float:
        """遍历所有行、列、对角线，为指定玩家累计评分。"""

        total = 0.0
        size = board.shape[0]

        for row in board:
            total += self._score_line(row.tolist(), player)

        for col in board.T:
            total += self._score_line(col.tolist(), player)

        for offset in range(-size + 1, size):
            diag = np.diagonal(board, offset)
            if diag.size >= 2:
                total += self._score_line(diag.tolist(), player)

        flipped = np.fliplr(board)
        for offset in range(-size + 1, size):
            diag = np.diagonal(flipped, offset)
            if diag.size >= 2:
                total += self._score_line(diag.tolist(), player)

        return total

    def _score_line(self, line: List[int], player: int) -> float:
        """计算一维序列中指定玩家的棋形得分。"""

        score = 0.0
        length = len(line)
        i = 0

        while i < length:
            if line[i] != player:
                i += 1
                continue

            j = i
            while j < length and line[j] == player:
                j += 1

            segment_len = j - i
            left_empty = 1 if i - 1 >= 0 and line[i - 1] == 0 else 0
            right_empty = 1 if j < length and line[j] == 0 else 0
            open_ends = left_empty + right_empty

            if segment_len >= 5:
                score += self._win_score
            else:
                score += self._pattern_weights.get((segment_len, open_ends), 0)

            i = j

        return score

    def _check_win(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        player: int,
    ) -> bool:
        """检查给定落子是否连成五子。"""

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        size = board.shape[0]

        for dr, dc in directions:
            count = 1

            nr, nc = row + dr, col + dc
            while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            nr, nc = row - dr, col - dc
            while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                return True

        return False

    def _has_five(self, board: np.ndarray, player: int) -> bool:
        """遍历整盘棋，判断指定玩家是否已经连五。"""

        positions = np.argwhere(board == player)
        for row, col in positions:
            if self._check_win(board, int(row), int(col), player):
                return True
        return False
