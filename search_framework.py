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

        # 评分常量
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

        # 搜索深度上限
        self._max_depth_upper = 7

        # 预留的数据结构
        self._zobrist_seed = 20240515
        self._zobrist_cache: Dict[int, np.ndarray] = {}
        self._zobrist_table: Optional[np.ndarray] = (
            None  # 当前棋盘尺寸对应的 Zobrist 哈希表
        )

        self._transposition_table: Dict[int, Dict[str, object]] = {}  # 置换表缓存
        self._transposition_limit = 200_000
        self._killer_moves: Dict[int, List[Tuple[int, int]]] = {}
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

        self._killer_moves.clear()
        if self._history_heuristic:
            for key in list(self._history_heuristic.keys()):
                self._history_heuristic[key] *= 0.9
                if self._history_heuristic[key] < 1.0:
                    del self._history_heuristic[key]

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
                ply=0,
            )

            if self._check_timeout(deadline):
                break

            info.update(
                {
                    "score": score,
                    "depth": depth,
                    "principal_variation": pv_line if pv_line else [],
                }
            )

            if move is not None:
                best_move = move
                best_score = score
                principal_variation = pv_line if pv_line else [move]

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
        ply: int = 0,
    ) -> Tuple[float, Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """Negamax + Alpha-Beta 剪枝的递归核心。"""

        if self._check_timeout(deadline):
            return self._evaluate_board(board_state), None, []

        if self._is_terminal(board_state):
            return self._evaluate_board(board_state), None, []

        if depth == 0:
            score = self._quiescence_search(alpha, beta, board_state, deadline, ply)
            return score, None, []

        best_score = float("-inf")
        best_move: Optional[Tuple[int, int]] = None
        best_line: List[Tuple[int, int]] = []

        alpha_initial = alpha
        beta_initial = beta

        tt_move_hint: Optional[Tuple[int, int]] = None
        tt_probe = self._probe_transposition(board_state, depth, alpha, beta)
        if tt_probe is not None:
            tt_value, tt_move, alpha, beta, tt_flag = tt_probe
            if tt_move is not None:
                tt_move_hint = tt_move
            if tt_flag in {"EXACT", "CUT"} and tt_value is not None:
                return tt_value, tt_move, [tt_move] if tt_move is not None else []

        moves = list(self._generate_moves(board_state))
        if not moves:
            return self._evaluate_board(board_state), None, []

        ordering_hint: List[Tuple[int, int]] = list(principal_variation)
        if tt_move_hint is not None and (
            not ordering_hint or ordering_hint[0] != tt_move_hint
        ):
            ordering_hint = [tt_move_hint, *ordering_hint]

        ordered_moves = self._order_moves(
            moves,
            board_state,
            ordering_hint,
            depth=depth,
            ply=ply,
        )

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
                ply=ply + 1,
            )

            score = -score

            self._undo_move(board_state)

            if score > best_score:
                best_score = score
                best_move = move
                best_line = [move, *child_line]

                if score > alpha_initial and move is not None:
                    self._update_history(board_state, move, depth, ply)

            alpha = max(alpha, best_score)
            if alpha >= beta:
                if move is not None:
                    self._register_killer_move(ply, move)
                break

            if self._check_timeout(deadline):
                break

        store_flag = "EXACT"
        if best_score <= alpha_initial:
            store_flag = "UPPER"
        elif best_score >= beta_initial:
            store_flag = "LOWER"

        self._store_transposition(
            board_state,
            depth,
            best_score,
            store_flag,
            best_move,
            alpha_initial,
            beta_initial,
        )

        return best_score, best_move, best_line

    def _quiescence_search(
        self,
        alpha: float,
        beta: float,
        board_state: Dict[str, object],
        deadline: float,
        ply: int = 0,
    ) -> float:
        """静态搜索：延伸强制胜负走法，缓解地平线效应。"""

        if self._check_timeout(deadline):
            return self._evaluate_board(board_state)

        stand_pat = self._evaluate_board(board_state)
        if stand_pat >= beta:
            return stand_pat

        if stand_pat > alpha:
            alpha = stand_pat

        urgent_moves: List[Tuple[int, int]] = []
        moves = list(self._generate_moves(board_state))
        player = board_state["current_player"]

        for move in moves:
            row, col = move
            board = board_state["board"]
            if board[row, col] != 0:
                continue
            board[row, col] = player
            is_win = self._check_win(board, row, col, player)
            board[row, col] = 0
            if is_win:
                urgent_moves.append(move)

        if not urgent_moves:
            return alpha

        urgent_moves = self._order_moves(
            urgent_moves,
            board_state,
            principal_variation=(),
            depth=0,
            ply=ply,
        )

        best = stand_pat

        for move in urgent_moves:
            self._apply_move(board_state, move)
            score = -self._quiescence_search(
                -beta, -alpha, board_state, deadline, ply + 1
            )
            self._undo_move(board_state)

            if score > best:
                best = score
            if best >= beta:
                self._register_killer_move(ply, move)
                return best
            if best > alpha:
                alpha = best

        return best

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
        depth: int = 0,
        ply: int = 0,
    ) -> List[Tuple[int, int]]:
        """结合多种启发式对走法按优先级排序。"""

        board = board_state["board"]
        player = board_state["current_player"]
        opponent = self._opponent(player)
        size = board_state["size"]
        center = (size - 1) / 2.0

        pv_set: Set[Tuple[int, int]] = {mv for mv in principal_variation}
        killer_moves = self._killer_moves.get(ply, [])
        scored_moves: List[Tuple[float, Tuple[int, int]]] = []

        for move in moves:
            row, col = move
            bonus = 0.0

            if move in pv_set:
                bonus += 10_000

            if move in killer_moves:
                # 最近的 killer 列表长度不超过 2
                bonus += 6_000 - killer_moves.index(move) * 500

            center_distance = abs(row - center) + abs(col - center)
            score = -center_distance

            hist_key = (player, row * size + col)
            history_score = self._history_heuristic.get(hist_key, 0.0)

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
            score += history_score
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
        # 当前回退为全量评估；若后续确认瓶颈，可实现局部增量更新。
        return self._evaluate_board(board_state)

    def _detect_patterns(
        self,
        board_array: np.ndarray,
    ) -> Dict[str, int]:
        """统计棋盘上的关键棋形，用于评分。"""
        # 预留丰富棋形检测的接口。
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

        previous_hash = board_state.get("hash", 0)

        record = {
            "move": move,
            "player": player,
            "previous_last_move": board_state.get("last_move"),
            "previous_last_player": board_state.get("last_player"),
            "previous_hash": previous_hash,
        }

        board_state["move_stack"].append(record)

        new_hash = self._update_hash(
            board_state, move, player, old_piece=board[row, col]
        )
        board[row, col] = player
        board_state["last_move"] = move
        board_state["last_player"] = player
        board_state["current_player"] = self._opponent(player)
        board_state["empty_count"] -= 1
        board_state["hash"] = new_hash

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
        board_state["hash"] = record.get("previous_hash", 0)

    # ------------------------------
    # 置换表与哈希
    # ------------------------------

    def _probe_transposition(
        self,
        board_state: Dict[str, object],
        depth: int,
        alpha: float,
        beta: float,
    ) -> Optional[Tuple[Optional[float], Optional[Tuple[int, int]], float, float, str]]:
        """置换表查询：返回缓存的分数与搜索窗口。"""

        key = board_state.get("hash")
        if key is None:
            return None

        entry = self._transposition_table.get(key)
        if entry is None:
            return None

        best_move = entry.get("best_move")
        entry_value = entry.get("value")
        entry_depth = entry.get("depth", 0)
        entry_flag = entry.get("flag", "NONE")

        if entry_depth < depth:
            return (None, best_move, alpha, beta, "HINT")

        if entry_flag == "EXACT":
            return (entry_value, best_move, alpha, beta, "EXACT")

        if entry_flag == "LOWER":
            alpha = max(alpha, entry_value)
        elif entry_flag == "UPPER":
            beta = min(beta, entry_value)

        if alpha >= beta:
            return (entry_value, best_move, alpha, beta, "CUT")

        return (None, best_move, alpha, beta, entry_flag)

    def _store_transposition(
        self,
        board_state: Dict[str, object],
        depth: int,
        score: float,
        flag: str,
        best_move: Optional[Tuple[int, int]],
        alpha: float,
        beta: float,
    ) -> None:
        """置换表写入：缓存已搜索的局面信息。"""

        key = board_state.get("hash")
        if key is None:
            return

        if len(self._transposition_table) >= self._transposition_limit:
            # 简单的 FIFO 淘汰策略
            try:
                self._transposition_table.pop(next(iter(self._transposition_table)))
            except StopIteration:
                pass

        self._transposition_table[key] = {
            "value": score,
            "depth": depth,
            "flag": flag,
            "best_move": best_move,
            "alpha": alpha,
            "beta": beta,
        }

    def _ensure_zobrist(self, board_size: int) -> None:
        """初始化或重置 Zobrist 哈希表。"""

        if board_size not in self._zobrist_cache:
            rng = np.random.default_rng(self._zobrist_seed + board_size)
            table = rng.integers(
                low=1,
                high=np.iinfo(np.uint64).max,
                size=(board_size, board_size, 3),
                dtype=np.uint64,
            )
            self._zobrist_cache[board_size] = table

        self._zobrist_table = self._zobrist_cache[board_size]

    def _update_hash(
        self,
        board_state: Dict[str, object],
        move: Tuple[int, int],
        player: int,
        old_piece: int = 0,
    ) -> int:
        """增量更新当前哈希值。"""

        if self._zobrist_table is None:
            raise RuntimeError("Zobrist 哈希表尚未初始化")

        current_hash = np.uint64(board_state.get("hash", 0))
        row, col = move

        if old_piece:
            current_hash ^= self._zobrist_table[row, col, old_piece]

        if player:
            current_hash ^= self._zobrist_table[row, col, player]

        return int(current_hash)

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

        board_hash = self._compute_hash(board_array)

        state: Dict[str, object] = {
            "board": board_array.copy(),
            "size": size,
            "current_player": current_player,
            "last_move": None,
            "last_player": None,
            "move_stack": [],
            "empty_count": empty_count,
            "hash": board_hash,
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
            return 5
        if empties > total * 0.3:
            return 6
        return min(self._max_depth_upper, 7)

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

    def _compute_hash(self, board_array: np.ndarray) -> int:
        """基于当前棋盘计算完整的 Zobrist 哈希值。"""

        if (
            self._zobrist_table is None
            or self._zobrist_table.shape[0] != board_array.shape[0]
        ):
            self._ensure_zobrist(board_array.shape[0])

        current_hash = np.uint64(0)
        positions = np.argwhere(board_array != 0)
        for row, col in positions:
            piece = int(board_array[row, col])
            current_hash ^= self._zobrist_table[row, col, piece]

        return int(current_hash)

    def _register_killer_move(self, ply: int, move: Tuple[int, int]) -> None:
        """记录在指定层深引发剪枝的 killer 走法。"""

        killers = self._killer_moves.setdefault(ply, [])
        if move in killers:
            killers.remove(move)
        killers.insert(0, move)
        if len(killers) > 2:
            killers.pop()

    def _update_history(
        self,
        board_state: Dict[str, object],
        move: Tuple[int, int],
        depth: int,
        ply: int,
    ) -> None:
        """根据提升 alpha 的走法更新历史启发表。"""

        size = board_state["size"]
        player = board_state["current_player"]
        key = (player, move[0] * size + move[1])
        bonus = depth * depth + max(0, 3 - ply)
        updated = self._history_heuristic.get(key, 0.0) + bonus
        self._history_heuristic[key] = min(updated, 1e7)
