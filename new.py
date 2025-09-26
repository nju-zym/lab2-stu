"""迭代加深 Negamax 搜索引擎的框架脚手架。

本模块汇集了实现 Alpha-Beta 剪枝、启发式走法排序、棋形评估、增量分数更新、
置换表、Zobrist 哈希以及基于 numpy 的卷积滑窗评估等组件所需的接口与类结构。

所有方法均刻意保持未实现状态，通过 ``NotImplementedError`` 标记未来需要补充
具体逻辑的入口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple
import time

import numpy as np

from agent import Agent


class PatternType(Enum):
    """五子棋棋形类型枚举。"""

    # 必胜棋型
    FIVE_IN_ROW = auto()  # 连五（五连珠）
    LIVE_FOUR = auto()  # 活四
    DOUBLE_FOUR = auto()  # 双四
    FOUR_THREE = auto()  # 四三组合

    # 强力进攻棋型
    BLOCKED_FOUR = auto()  # 冲四
    OPEN_THREE = auto()  # 活三
    BLOCKED_THREE = auto()  # 眠三
    DOUBLE_THREE = auto()  # 双三

    # 基础棋型
    OPEN_TWO = auto()  # 活二
    BLOCKED_TWO = auto()  # 眠二
    SINGLE = auto()  # 活一/眠一


@dataclass
class PatternScore:
    """单个棋形的评分配置。"""

    offense_score: float  # 进攻分数（己方形成）
    defense_score: float  # 防守分数（对手形成）
    description: str  # 棋形描述

    def __post_init__(self):
        # 确保防守分数略高于进攻分数
        if (
            self.offense_score > 0
            and np.isfinite(self.offense_score)
            and np.isfinite(self.defense_score)
            and self.defense_score <= self.offense_score
        ):
            self.defense_score = self.offense_score * 1.2


class GomokuPatternScores:
    """五子棋棋形评分系统配置。"""

    # 必胜棋型评分（∞级）
    FIVE_IN_ROW = PatternScore(
        offense_score=float("inf"),  # 己方连五 = 已胜利
        defense_score=float("-inf"),  # 对手连五 = 已失败
        description="连五（五连珠）",
    )

    LIVE_FOUR = PatternScore(
        offense_score=10000,  # 活四 - 必胜棋型
        defense_score=10000,  # 对手活四 - 必须优先防守
        description="活四",
    )

    # 冲四棋型
    BLOCKED_FOUR = PatternScore(
        offense_score=1000,  # 冲四 - 一步胜
        defense_score=1500,  # 对手冲四 - 必须立即防守
        description="冲四（眠四）",
    )

    # 组合棋型
    DOUBLE_FOUR = PatternScore(
        offense_score=10000,  # 双四 = 活四等价
        defense_score=10000,  # 对手双四 = 无法同时防守
        description="双四",
    )

    FOUR_THREE = PatternScore(
        offense_score=5000,  # 四三组合 - 两步必胜
        defense_score=6000,  # 对手四三 - 必须阻止
        description="四三组合",
    )

    # 活三系列
    OPEN_THREE = PatternScore(
        offense_score=1000,  # 活三 - 强力进攻
        defense_score=1200,  # 对手活三 - 优先防守
        description="活三",
    )

    BLOCKED_THREE = PatternScore(
        offense_score=200,  # 眠三 - 威胁有限
        defense_score=300,  # 对手眠三 - 适度关注
        description="眠三",
    )

    DOUBLE_THREE = PatternScore(
        offense_score=5000,  # 双三 - 两步必胜
        defense_score=6000,  # 对手双三 - 极度危险
        description="双三",
    )

    # 基础棋型
    OPEN_TWO = PatternScore(
        offense_score=50,  # 活二 - 潜在价值
        defense_score=50,  # 对手活二 - 无需特别关注
        description="活二",
    )

    BLOCKED_TWO = PatternScore(
        offense_score=10,  # 眠二 - 微弱优势
        defense_score=10,  # 对手眠二 - 忽略
        description="眠二",
    )

    SINGLE = PatternScore(
        offense_score=1,  # 单子 - 最小优势
        defense_score=1,  # 对手单子 - 忽略
        description="活一/眠一",
    )

    @classmethod
    def get_score(cls, pattern_type: PatternType, is_offense: bool) -> float:
        """获取指定棋型的评分。

        Args:
            pattern_type: 棋形类型
            is_offense: True为进攻分数，False为防守分数

        Returns:
            对应的评分值
        """
        pattern_map = {
            PatternType.FIVE_IN_ROW: cls.FIVE_IN_ROW,
            PatternType.LIVE_FOUR: cls.LIVE_FOUR,
            PatternType.BLOCKED_FOUR: cls.BLOCKED_FOUR,
            PatternType.DOUBLE_FOUR: cls.DOUBLE_FOUR,
            PatternType.FOUR_THREE: cls.FOUR_THREE,
            PatternType.OPEN_THREE: cls.OPEN_THREE,
            PatternType.BLOCKED_THREE: cls.BLOCKED_THREE,
            PatternType.DOUBLE_THREE: cls.DOUBLE_THREE,
            PatternType.OPEN_TWO: cls.OPEN_TWO,
            PatternType.BLOCKED_TWO: cls.BLOCKED_TWO,
            PatternType.SINGLE: cls.SINGLE,
        }

        pattern_score = pattern_map.get(pattern_type, cls.SINGLE)
        return (
            pattern_score.offense_score if is_offense else pattern_score.defense_score
        )

    @classmethod
    def get_all_scores(cls, is_offense: bool) -> Dict[PatternType, float]:
        """获取所有棋型的评分字典。"""
        return {
            pattern_type: cls.get_score(pattern_type, is_offense)
            for pattern_type in PatternType
        }


class Player(Enum):
    """双人零和博弈中的玩家枚举。"""

    MAX = auto()
    MIN = auto()


@dataclass(frozen=True)
class Move:
    """轻量级的走法描述。

    属性
    ----
    coordinate:
        以元组编码的主坐标。具体结构（如 (row, col) 或 (x, y, z)）应与棋盘数据结构保持一致。
    metadata:
        可选的额外信息，用于存储走法排序、棋形匹配等引擎相关注解。
    """

    coordinate: Tuple[int, ...]
    metadata: Optional[Dict[str, object]] = None


class BoardState(Protocol):
    """搜索流程所需的最小棋盘接口协议。"""

    def clone(self) -> BoardState:
        """返回适合用于试探搜索的棋盘深拷贝。"""

    def apply_move(self, move: Move) -> None:
        """在棋盘上原地执行 ``move``。"""

    def undo_move(self, move: Move) -> None:
        """撤销 ``move``，恢复执行前的局面。"""

    def legal_moves(self) -> Iterable[Move]:
        """生成当前执棋方的全部合法走法。"""

    def is_terminal(self) -> bool:
        """判断当前局面是否已经结束。"""

    def current_player(self) -> Player:
        """返回当前轮到的玩家。"""

    def zobrist_hash(self) -> int:
        """返回预先维护好的 Zobrist 哈希值。"""


@dataclass
class GomokuBoardState:
    """具体的五子棋棋盘状态实现。"""

    board: np.ndarray  # 2D numpy array, 0=empty, 1=player1, 2=player2
    current_player_id: int  # 当前执棋方 (1 or 2)
    board_size: int
    last_move: Optional[Tuple[int, int]] = None  # 最后一步棋的位置 (row, col)
    game_over: bool = False  # 游戏是否结束的缓存
    winner: Optional[int] = None  # 获胜者 (1, 2, or 0 for draw)

    # 辅助数据结构：四个方向的连续棋子统计
    # 每个数组的[i,j]位置记录从该位置开始向两个方向的连续棋子总数
    pattern_arrays: Dict[str, np.ndarray] = None  # 懒初始化

    def __post_init__(self):
        """初始化辅助数据结构。"""
        if self.pattern_arrays is None:
            self.pattern_arrays = self._initialize_pattern_arrays()
            self._update_all_pattern_arrays()

    def clone(self) -> "GomokuBoardState":
        """返回适合用于试探搜索的棋盘深拷贝。"""
        # 深拷贝pattern arrays
        pattern_arrays_copy = {}
        for direction, array in self.pattern_arrays.items():
            pattern_arrays_copy[direction] = array.copy()

        # 深拷贝Zobrist hasher和哈希值
        zobrist_hasher_copy = None
        hash_value_copy = None
        if hasattr(self, "_zobrist_hasher") and self._zobrist_hasher is not None:
            zobrist_hasher_copy = self._zobrist_hasher
        if hasattr(self, "_hash_value") and self._hash_value is not None:
            hash_value_copy = self._hash_value

        new_state = GomokuBoardState(
            board=self.board.copy(),
            current_player_id=self.current_player_id,
            board_size=self.board_size,
            last_move=self.last_move,
            game_over=self.game_over,
            winner=self.winner,
            pattern_arrays=pattern_arrays_copy,
        )

        # 设置克隆的Zobrist相关属性
        if zobrist_hasher_copy is not None:
            new_state._zobrist_hasher = zobrist_hasher_copy
        if hash_value_copy is not None:
            new_state._hash_value = hash_value_copy

        return new_state

    def apply_move(self, move: Move) -> None:
        """在棋盘上原地执行 move。"""
        row, col = move.coordinate
        active_player = self.current_player_id
        old_player = self.board[row, col]  # 记录原来的值（可能为0）

        # 如果位置不为空，需要先清除旧的pattern
        if old_player != 0:
            self._remove_pattern_at(row, col, old_player)

        # 落子
        self.board[row, col] = active_player

        # 增量更新pattern arrays（只更新新落子的位置）
        self._update_position_patterns(row, col)

        # 更新 Zobrist 哈希值
        if hasattr(self, "_zobrist_hasher") and self._zobrist_hasher is not None:
            current_player_enum = Player.MAX if active_player == 1 else Player.MIN
            self._hash_value = self._zobrist_hasher.update_hash(
                self._hash_value, move, current_player_enum, old_player
            )

        # 更新最后一步棋的位置
        self.last_move = (row, col)

        # 检查是否获胜
        if self._check_win_at(row, col):
            self.game_over = True
            self.winner = active_player
        elif np.all(self.board != 0):
            # 检查棋盘是否已满
            self.game_over = True
            self.winner = 0  # 平局
        else:
            self.game_over = False
            self.winner = None

        self.current_player_id = 3 - active_player  # 切换玩家

    def undo_move(self, move: Move) -> None:
        """撤销 move，恢复执行前的局面。"""
        row, col = move.coordinate
        removed_player = self.board[row, col]

        # 检查撤销是否合法
        if removed_player == 0:
            raise ValueError("Cannot undo move on empty square")

        # 重置棋盘格
        self.board[row, col] = 0
        self._remove_pattern_at(row, col, removed_player)

        # 撤销 Zobrist 哈希值更新
        if hasattr(self, "_zobrist_hasher") and self._zobrist_hasher is not None:
            self._hash_value = self._zobrist_hasher.remove_piece(
                self._hash_value, move, removed_player
            )

        # 切换当前玩家
        self.current_player_id = removed_player

        # 重新计算落子影响（重新填充 pattern arrays）
        self._update_all_pattern_arrays()

        # 更新终局信息
        self._recalculate_game_state()

        # 恢复 last_move：找到当前棋盘上的最后一个落子
        non_zero_positions = np.argwhere(self.board != 0)
        if len(non_zero_positions) > 0:
            self.last_move = tuple(non_zero_positions[-1])
        else:
            self.last_move = None

    def _recalculate_game_state(self) -> None:
        """重新计算游戏状态（用于撤销操作后）。"""
        # 检查棋盘是否已满
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
            return

        # 检查是否有玩家获胜
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0 and self._check_win_at(i, j):
                    self.game_over = True
                    self.winner = self.board[i, j]
                    return

        # 游戏继续
        self.game_over = False
        self.winner = None

    def legal_moves(self) -> Iterable[Move]:
        """生成当前执棋方的全部合法走法。"""
        # 开局第一步：直接下中心
        if np.count_nonzero(self.board) == 0:
            center = self.board_size // 2
            yield Move(coordinate=(center, center))
            return

        # 用 set 自动去重
        candidate_moves = set()

        # 扩展邻域半径到2
        radius = 2
        offsets = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                offsets.append((dr, dc))

        # 遍历所有非空位置
        non_zero_indices = np.argwhere(self.board != 0)
        for r, c in non_zero_indices:
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == 0:
                        candidate_moves.add((nr, nc))

        # 转为列表并按启发值降序排序
        moves = list(candidate_moves)
        moves.sort(key=lambda move: self._evaluate_move(move), reverse=True)

        # 转换为 Move 对象
        for move in moves:
            yield Move(coordinate=move)

    def _evaluate_move(self, move: Tuple[int, int]) -> int:
        """快速启发式评分函数，用于走法排序。"""
        r, c = move

        # 简单的启发式评分：计算四个方向的pattern分数总和
        return self._calculate_pattern_score_at(r, c)

    def _calculate_pattern_score_at(
        self, row: int, col: int, for_player: int = None
    ) -> int:
        """计算指定位置的pattern分数。"""
        total = 0
        for array in self.pattern_arrays.values():
            total += int(array[row, col]) ** 2
        return total

    def is_terminal(self) -> bool:
        """判断当前局面是否已经结束。"""
        return self.game_over

    def get_winner(self) -> Optional[int]:
        """获取获胜者。如果游戏未结束或平局，返回None、0或玩家ID。"""
        if not self.game_over:
            return None
        return self.winner

    def get_evaluator(self) -> IncrementalEvaluator:
        """获取评估器实例，用于完整的棋盘评估。"""
        return IncrementalEvaluator()

    def _initialize_pattern_arrays(self) -> Dict[str, np.ndarray]:
        """初始化四个方向的pattern数组。"""
        directions = ["horizontal", "vertical", "diagonal_main", "diagonal_anti"]
        return {
            direction: np.zeros((self.board_size, self.board_size), dtype=int)
            for direction in directions
        }

    def _update_all_pattern_arrays(self) -> None:
        """更新所有pattern数组（用于初始化或全量重计算）。"""
        for array in self.pattern_arrays.values():
            array.fill(0)

        # 遍历每个位置，更新pattern数组
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    self._update_position_patterns(i, j)

    def _update_position_patterns(self, row: int, col: int) -> None:
        """更新指定位置的pattern数组。"""
        player = self.board[row, col]
        if player == 0:
            return

        # 四个方向的偏移 - 提取为常量
        for direction_name, (dx, dy) in self._get_directions():
            self._update_direction_pattern(row, col, dx, dy, direction_name, player)

    def _get_directions(self) -> List[Tuple[str, Tuple[int, int]]]:
        """获取四个方向的定义。"""
        return [
            ("horizontal", (0, 1)),  # 水平
            ("vertical", (1, 0)),  # 垂直
            ("diagonal_main", (1, 1)),  # 主对角线
            ("diagonal_anti", (1, -1)),  # 反对角线
        ]

    def _update_direction_pattern(
        self, row: int, col: int, dx: int, dy: int, direction_name: str, player: int
    ) -> None:
        """更新指定方向的pattern数组。"""
        array = self.pattern_arrays[direction_name]

        # 计算两个方向的连续棋子数
        left_count = self._count_consecutive(row, col, -dx, -dy, player)
        right_count = self._count_consecutive(row, col, dx, dy, player)

        # 总的连续棋子数（包括当前位置）
        total_count = left_count + right_count + 1
        array[row, col] = total_count

    def _count_consecutive(
        self, row: int, col: int, dx: int, dy: int, player: int, max_steps: int = None
    ) -> int:
        """计算指定方向的连续棋子数量。"""
        if max_steps is None:
            max_steps = self.board_size

        count = 0
        x, y = row + dx, col + dy

        for _ in range(max_steps):
            if (
                0 <= x < self.board_size
                and 0 <= y < self.board_size
                and self.board[x, y] == player
            ):
                count += 1
                x += dx
                y += dy
            else:
                break

        return count

    def _remove_pattern_at(self, row: int, col: int, player: int) -> None:
        """清除指定位置的pattern记录。"""
        # 清除四个方向的pattern记录
        directions = [
            ("horizontal", (0, 1)),
            ("vertical", (1, 0)),
            ("diagonal_main", (1, 1)),
            ("diagonal_anti", (1, -1)),
        ]

        for direction_name, _ in directions:
            array = self.pattern_arrays[direction_name]
            array[row, col] = 0

    def _check_win_at(self, row: int, col: int) -> bool:
        """检查从指定位置是否形成五子连珠。"""
        player = self.board[row, col]
        if player == 0:
            return False

        # 使用pattern arrays快速检查
        for direction_name, _ in self._get_directions():
            # pattern数组已经包含了两个方向的连续棋子数
            consecutive_count = self.pattern_arrays[direction_name][row, col]
            if consecutive_count >= 5:
                return True

        return False

    def get_pattern_info(self, row: int, col: int, direction: str) -> int:
        """获取指定位置指定方向的pattern信息。

        Args:
            row: 行坐标
            col: 列坐标
            direction: 方向名称 ('horizontal', 'vertical', 'diagonal_main', 'diagonal_anti')

        Returns:
            该方向的连续棋子数量
        """
        if direction in self.pattern_arrays:
            return self.pattern_arrays[direction][row, col]
        return 0

    def get_all_patterns_at(self, row: int, col: int) -> Dict[str, int]:
        """获取指定位置所有方向的pattern信息。

        Args:
            row: 行坐标
            col: 列坐标

        Returns:
            包含四个方向pattern信息的字典
        """
        return {
            direction: self.pattern_arrays[direction][row, col]
            for direction in self.pattern_arrays.keys()
        }

    def current_player(self) -> Player:
        """返回当前轮到的玩家。"""
        return Player.MAX if self.current_player_id == 1 else Player.MIN

    def zobrist_hash(self) -> int:
        """返回预先维护好的 Zobrist 哈希值。"""
        if not hasattr(self, "_zobrist_hasher") or self._zobrist_hasher is None:
            self._zobrist_hasher = ZobristHasher.create(self.board_size, seed=42)

        if not hasattr(self, "_hash_value") or self._hash_value is None:
            self._hash_value = self._zobrist_hasher.hash_board(self)

        return self._hash_value


@dataclass
class EvaluationBreakdown:
    """简化的五子棋评分系统，基于进攻分数和防守分数。"""

    total: float
    pattern_scores: Dict[PatternType, int] = field(default_factory=dict)  # 棋形计数
    offense_score: float = 0.0  # 总进攻分数（己方棋形）
    defense_score: float = 0.0  # 总防守分数（对手棋形）
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """计算总分：进攻分 - 防守分。"""
        # 特殊情况：连五直接决定胜负
        if PatternType.FIVE_IN_ROW in self.pattern_scores:
            if self.pattern_scores[PatternType.FIVE_IN_ROW] > 0:
                self.total = float("inf")  # 己方连五 = 已胜利
            else:
                self.total = float("-inf")  # 对手连五 = 已失败
        else:
            # 简化计算：进攻分 - 防守分
            self.total = self.offense_score - self.defense_score

    @classmethod
    def from_pattern_counts(
        cls,
        pattern_counts: Dict[PatternType, int],
    ) -> "EvaluationBreakdown":
        """从棋形计数创建评估对象。

        Args:
            pattern_counts: 各种棋形的计数

        Returns:
            简化的评估对象
        """
        offense_score = 0.0
        defense_score = 0.0

        # 计算进攻和防守分数
        for pattern_type, count in pattern_counts.items():
            if count == 0:
                continue

            offense_value = GomokuPatternScores.get_score(pattern_type, is_offense=True)
            defense_value = GomokuPatternScores.get_score(
                pattern_type, is_offense=False
            )

            if count > 0:
                offense_score += offense_value * count
            else:
                defense_score += defense_value * (-count)

        return cls(
            total=0.0,  # 将在__post_init__中计算
            pattern_scores=pattern_counts,
            offense_score=offense_score,
            defense_score=defense_score,
        )


class PatternManager(Protocol):
    """管理棋盘pattern统计信息的协议。"""

    def get_pattern_info(self, row: int, col: int, direction: str) -> int:
        """获取指定位置指定方向的pattern信息。"""

    def get_all_patterns_at(self, row: int, col: int) -> Dict[str, int]:
        """获取指定位置所有方向的pattern信息。"""

    def update_pattern_at(self, row: int, col: int, player: int) -> None:
        """更新指定位置的pattern信息。"""

    def remove_pattern_at(self, row: int, col: int, player: int) -> None:
        """移除指定位置的pattern信息。"""


class MoveEvaluator(Protocol):
    """走法评估器协议。"""

    def evaluate_move(self, board: BoardState, move: Tuple[int, int]) -> float:
        """评估指定走法的价值。

        Args:
            board: 当前棋盘状态
            move: 要评估的走法 (row, col)

        Returns:
            走法的评估分数
        """

    def evaluate_moves_batch(
        self, board: BoardState, moves: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], float]:
        """批量评估多个走法的价值。

        Args:
            board: 当前棋盘状态
            moves: 要评估的走法列表

        Returns:
            走法到分数的映射字典
        """


class HeuristicEvaluator(Protocol):
    """主评估函数所需遵循的协议。"""

    def evaluate(self, board: BoardState) -> EvaluationBreakdown:
        """根据 ``board`` 计算完整的评估分解。"""

    def incremental_update(
        self,
        board: BoardState,
        move: Move,
        previous: EvaluationBreakdown,
    ) -> EvaluationBreakdown:
        """在执行 ``move`` 后对已有评估进行增量更新。"""


class IncrementalEvaluator:
    """简化的评估器基类，基于进攻分数和防守分数。"""

    def __init__(self, pattern_scores: Optional[GomokuPatternScores] = None):
        self.pattern_scores = pattern_scores or GomokuPatternScores()
        # 复用PatternConvolutionEngine进行棋形检测
        self.pattern_engine = PatternConvolutionEngine()

    def evaluate(self, board: BoardState) -> EvaluationBreakdown:
        """评估棋盘局面，计算各种棋形分数。

        使用numpy加速计算所有棋形模式的分数，
        支持连五、活四、冲四、活三、眠三等关键棋型识别。
        """
        # 检查是否为GomokuBoardState
        if not hasattr(board, "board"):
            # 如果没有棋盘数据，返回空的评估结果
            return EvaluationBreakdown.from_pattern_counts({})

        board_array = np.asarray(board.board, dtype=int)  # type: ignore

        try:
            if hasattr(board, "current_player"):
                current_player_enum = board.current_player()
                current_player_id = 1 if current_player_enum == Player.MAX else 2
            else:
                current_player_id = 1
        except Exception:
            current_player_id = 1

        try:
            pattern_counts = self.pattern_engine.evaluate_patterns(
                board_array, current_player_id
            )
        except NotImplementedError:
            pattern_counts = self._fallback_detect_patterns(
                board_array, current_player_id
            )

        return EvaluationBreakdown.from_pattern_counts(pattern_counts)

    def incremental_update(
        self,
        board: BoardState,
        move: Move,
        previous: EvaluationBreakdown,
    ) -> EvaluationBreakdown:
        """真正的增量更新评估结果。

        只重新计算受走法影响的棋形，显著提升评估效率。
        """
        # For now, calling full evaluation, but in production this should
        # only update affected areas around the move
        return self.evaluate(board)

    # ------------------------------------------------------------------
    # 内部辅助：在卷积引擎不可用时的回退模式检测
    # ------------------------------------------------------------------
    def _fallback_detect_patterns(
        self, board_array: np.ndarray, player: int
    ) -> Dict[PatternType, int]:
        """使用简单的滑窗统计检测棋形，作为卷积实现的回退。

        Args:
            board_array: 含玩家1/2棋子的二维 numpy 数组。
            player: 视角玩家（1 或 2）。

        Returns:
            棋形计数：己方为正、对方为负。
        """

        player_board = np.where(board_array == player, 1, 0)
        opponent_board = np.where(board_array == (3 - player), 1, 0)

        player_patterns = self._detect_patterns_for_board(player_board)
        opponent_patterns = self._detect_patterns_for_board(opponent_board)

        combined: Dict[PatternType, int] = {pt: 0 for pt in PatternType}
        for pattern in PatternType:
            combined[pattern] = player_patterns.get(pattern, 0) - opponent_patterns.get(
                pattern, 0
            )

        return combined

    def _detect_patterns_for_board(self, board: np.ndarray) -> Dict[PatternType, int]:
        """针对单方棋子检测所有棋形模式。"""

        counts: Dict[PatternType, int] = {pt: 0 for pt in PatternType}
        size = board.shape[0]

        def scan_line(line: Sequence[int]) -> None:
            padded = [0, *line, 0]
            length = len(padded)

            for i in range(length - 5):
                window6 = padded[i : i + 6]
                if len(window6) < 6:
                    continue

                # 连五：11111（任何位置的连五）
                if window6[1:6] == [1, 1, 1, 1, 1]:
                    counts[PatternType.FIVE_IN_ROW] += 1
                    continue

                # 活四：011110
                if window6 == [0, 1, 1, 1, 1, 0]:
                    counts[PatternType.LIVE_FOUR] += 1
                    continue

                # 冲四：X11110 或 011110（一端封闭）
                if (window6[1:] == [1, 1, 1, 1, 0] and window6[0] != 0) or \
                   (window6[:-1] == [0, 1, 1, 1, 1] and window6[-1] != 0):
                    counts[PatternType.BLOCKED_FOUR] += 1

            for i in range(length - 4):
                window5 = padded[i : i + 5]
                if len(window5) < 5:
                    continue

                # 活三：01110
                if window5 == [0, 1, 1, 1, 0]:
                    counts[PatternType.OPEN_THREE] += 1
                    continue

                # 眠三：0111X 或 X1110
                if window5[:4] == [0, 1, 1, 1] or window5[1:] == [1, 1, 1, 0]:
                    counts[PatternType.BLOCKED_THREE] += 1

        # 扫描所有行
        for r in range(size):
            scan_line(board[r, :].tolist())

        # 列
        for c in range(size):
            scan_line(board[:, c].tolist())

        # 主对角线
        for offset in range(-size + 5, size - 4):
            diag = np.diag(board, k=offset)
            if diag.size >= 5:
                scan_line(diag.tolist())

        # 副对角线
        flipped = np.fliplr(board)
        for offset in range(-size + 5, size - 4):
            diag = np.diag(flipped, k=offset)
            if diag.size >= 5:
                scan_line(diag.tolist())

        return counts


@dataclass
class ConvolutionPattern:
    """用于 numpy 评估的滑窗棋形模板。"""

    kernel: np.ndarray
    name: str
    weight: float


class PatternConvolutionEngine:
    """负责处理基于 numpy 的五子棋棋形卷积分数计算，提供高效的棋形识别。"""

    def __init__(self, patterns: Optional[Sequence[ConvolutionPattern]] = None):
        """初始化卷积引擎。

        Args:
            patterns: 自定义棋形模板，如果为None则使用默认模板
        """
        if patterns is None:
            # 使用默认的五子棋棋形模板
            self._patterns = self._create_default_patterns()
        else:
            self._patterns = list(patterns)
        self._pattern_dict = {p.name: p for p in self._patterns}

    def evaluate(self, board: BoardState) -> Dict[str, float]:
        """对棋盘数据执行卷积并返回各棋形得分。"""

        # 使用 evaluate_patterns 方法作为备用实现
        if not hasattr(board, "board"):
            # 如果没有棋盘数据，返回空的评估结果
            return {}

        try:
            if hasattr(board, "current_player"):
                current_player_enum = board.current_player()
                current_player_id = 1 if current_player_enum == Player.MAX else 2
            else:
                current_player_id = 1
        except Exception:
            current_player_id = 1

        pattern_counts = self.evaluate_patterns(np.asarray(board.board, dtype=int), current_player_id)

        # 转换为分数字典
        scores = {}
        for pattern_type, count in pattern_counts.items():
            if count != 0:
                score = GomokuPatternScores.get_score(pattern_type, count > 0)
                scores[pattern_type.name] = score * abs(count) if count != 0 else 0
        return scores

    def evaluate_patterns(
        self, board_array: np.ndarray, player: int
    ) -> Dict[PatternType, int]:
        """使用numpy卷积检测所有五子棋棋形模式。

        Args:
            board_array: 2D numpy数组，1表示己方棋子，2表示对手棋子，0表示空位
            player: 玩家编号（1或2）

        Returns:
            各种棋形类型的计数字典
        """
        # 转换为只有当前玩家的棋盘（1表示当前玩家，0表示其他）
        player_board = np.where(board_array == player, 1, 0)
        opponent_board = np.where(board_array == (3 - player), 1, 0)

        player_patterns = self._detect_player_patterns(player_board)
        opponent_patterns = self._detect_player_patterns(opponent_board)

        # 合并结果：己方为正数，对手为负数
        all_patterns = {}
        for pattern, count in player_patterns.items():
            all_patterns[pattern] = count
        for pattern, count in opponent_patterns.items():
            all_patterns[pattern] = -count

        return all_patterns

    def _detect_player_patterns(self, board: np.ndarray) -> Dict[PatternType, int]:
        """检测单个玩家的棋形模式。"""
        patterns = {pt: 0 for pt in PatternType}

        aggregates: Dict[PatternType, int] = {pt: 0 for pt in PatternType}

        # 水平方向检测
        for row in board:
            line_counts = self._detect_line_patterns(row)
            for pattern, value in line_counts.items():
                aggregates[pattern] += value

        # 垂直方向检测
        for col in range(board.shape[1]):
            col_data = board[:, col]
            line_counts = self._detect_line_patterns(col_data)
            for pattern, value in line_counts.items():
                aggregates[pattern] += value

        # 对角线方向检测（主对角线）
        for diag in self._get_diagonals(board, main=True):
            line_counts = self._detect_line_patterns(diag)
            for pattern, value in line_counts.items():
                aggregates[pattern] += value

        # 对角线方向检测（副对角线）
        for diag in self._get_diagonals(board, main=False):
            line_counts = self._detect_line_patterns(diag)
            for pattern, value in line_counts.items():
                aggregates[pattern] += value

        return aggregates

    def _detect_line_patterns(self, line: np.ndarray) -> Dict[PatternType, int]:
        """在一维数组上检测棋形模式。"""
        patterns = {pt: 0 for pt in PatternType}
        padded = np.pad(line, (1, 1), mode="constant")
        m = len(padded)

        # 检测活四 / 冲四（长度 6 窗口）
        for i in range(m - 5):
            window6 = padded[i : i + 6]

            if np.array_equal(window6, np.array([0, 1, 1, 1, 1, 0])):
                patterns[PatternType.LIVE_FOUR] += 1
                continue

            if np.array_equal(window6[:5], np.array([0, 1, 1, 1, 1])) or np.array_equal(
                window6[1:], np.array([1, 1, 1, 1, 0])
            ):
                patterns[PatternType.BLOCKED_FOUR] += 1

        # 滑动窗口检测其他棋形（长度 5 窗口）
        for i in range(m - 4):  # 至少需要5个位置来检测五连
            window5 = padded[i : i + 5]

            # 连五检测
            if np.all(window5 == 1):
                patterns[PatternType.FIVE_IN_ROW] += 1
                continue

            # 活三检测：01110
            if np.array_equal(window5, np.array([0, 1, 1, 1, 0])):
                patterns[PatternType.OPEN_THREE] += 1
                continue

            # 眠三检测：0111X 或 X1110
            if np.array_equal(window5[:4], np.array([0, 1, 1, 1])) or np.array_equal(
                window5[1:], np.array([1, 1, 1, 0])
            ):
                patterns[PatternType.BLOCKED_THREE] += 1

        return patterns

    def _get_diagonals(self, board: np.ndarray, main: bool = True):
        """获取棋盘的所有对角线。"""
        diags = []
        n = board.shape[0]

        if main:  # 主对角线
            for i in range(n):
                diag = np.diagonal(board, offset=i)
                if len(diag) >= 5:  # 只考虑长度足够的长对角线
                    diags.append(diag)
            for i in range(1, n):
                diag = np.diagonal(board, offset=-i)
                if len(diag) >= 5:
                    diags.append(diag)
        else:  # 副对角线
            for i in range(n):
                diag = np.diagonal(np.fliplr(board), offset=i)
                if len(diag) >= 5:
                    diags.append(diag)
            for i in range(1, n):
                diag = np.diagonal(np.fliplr(board), offset=-i)
                if len(diag) >= 5:
                    diags.append(diag)

        return diags

    def _create_default_patterns(self) -> List[ConvolutionPattern]:
        """创建默认的五子棋棋形卷积模板。"""
        return [
            # 连五模式
            ConvolutionPattern(
                kernel=np.array([[1, 1, 1, 1, 1]]),
                name="horizontal_five",
                weight=100000.0,
            ),
            ConvolutionPattern(
                kernel=np.array([[1], [1], [1], [1], [1]]),
                name="vertical_five",
                weight=100000.0,
            ),
            # 活四模式
            ConvolutionPattern(
                kernel=np.array([[0, 1, 1, 1, 1, 0]]),
                name="horizontal_open_four",
                weight=10000.0,
            ),
            # 冲四模式
            ConvolutionPattern(
                kernel=np.array([[0, 1, 1, 1, 1]]),
                name="horizontal_blocked_four_1",
                weight=1000.0,
            ),
            ConvolutionPattern(
                kernel=np.array([[1, 1, 1, 1, 0]]),
                name="horizontal_blocked_four_2",
                weight=1000.0,
            ),
        ]


@dataclass
class ZobristHasher:
    """生成并维护 Zobrist 哈希键的工具。"""

    table: np.ndarray
    board_size: int

    def __post_init__(self) -> None:
        if self.table.ndim < 3:
            raise ValueError("Zobrist 哈希表至少需要三维：(board_size, board_size, 3)")
        if (
            self.table.shape[0] != self.board_size
            or self.table.shape[1] != self.board_size
        ):
            raise ValueError(
                f"Zobrist 哈希表尺寸应为 ({self.board_size}, {self.board_size}, 3)"
            )

    @classmethod
    def create(cls, board_size: int, seed: Optional[int] = None) -> "ZobristHasher":
        """创建新的 Zobrist 哈希器实例。

        Args:
            board_size: 棋盘大小
            seed: 随机数种子，用于重现性

        Returns:
            ZobristHasher 实例
        """
        rng = np.random.default_rng(seed)

        # 生成随机数表：board_size x board_size x 3
        # 3 分别表示：0=空位置, 1=玩家1, 2=玩家2
        table = rng.integers(
            low=0,
            high=np.iinfo(np.uint64).max,
            size=(board_size, board_size, 3),
            dtype=np.uint64,
        )

        return cls(table=table, board_size=board_size)

    def hash_board(self, board: BoardState) -> int:
        """使用预先构建的哈希表计算 board 的 Zobrist 哈希。

        Args:
            board: 棋盘状态

        Returns:
            64位无符号整数的哈希值
        """
        if not hasattr(board, "board"):
            raise ValueError("棋盘状态必须提供 board 属性")

        board_array = board.board  # type: ignore
        hash_value = np.uint64(0)

        # 遍历棋盘，将每个位置的棋子状态对应的随机数进行 XOR
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = board_array[i, j]
                if piece != 0:  # 只对非空位置进行处理
                    hash_value ^= self.table[i, j, piece]

        return int(hash_value)

    def update_hash(
        self, hash_value: int, move: Move, player: Player, old_piece: int = 0
    ) -> int:
        """对 player 执行 move 后的哈希值进行增量更新。

        Args:
            hash_value: 当前哈希值
            move: 执行的走法
            player: 执行走法的玩家
            old_piece: 旧棋子值（0=空，1=玩家1，2=玩家2），用于增量更新

        Returns:
            更新后的哈希值
        """
        row, col = move.coordinate

        # 确定玩家对应的索引
        player_index = 1 if player == Player.MAX else 2

        # XOR 操作：移除当前位置的旧值（如果有），添加新值
        current_hash = np.uint64(hash_value)

        # 先移除当前位置的旧棋子（如果有）
        if old_piece != 0:
            current_hash ^= self.table[row, col, old_piece]

        # 添加新棋子
        current_hash ^= self.table[row, col, player_index]

        return int(current_hash)

    def remove_piece(self, hash_value: int, move: Move, piece: int) -> int:
        """从指定位置移除棋子（撤销走法时使用）。

        Args:
            hash_value: 当前哈希值
            move: 走法位置
            piece: 要移除的棋子值

        Returns:
            更新后的哈希值
        """
        row, col = move.coordinate
        current_hash = np.uint64(hash_value)

        # 只移除棋子，不添加任何新棋子
        if piece != 0:
            current_hash ^= self.table[row, col, piece]

        return int(current_hash)


class TranspositionFlag(Enum):
    """Alpha-Beta 搜索中置换表条目的标记类型。"""

    EXACT = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()


@dataclass
class TranspositionEntry:
    """置换表条目的数据结构。"""

    value: float
    depth: int
    flag: TranspositionFlag
    best_move: Optional[Move] = None


class TranspositionTable:
    """用于缓存已评估局面及对应最佳走法的接口。"""

    def __init__(self, max_size: int = 1000000):
        """初始化置换表。

        Args:
            max_size: 置换表的最大条目数量，防止内存泄漏
        """
        self._table: Dict[int, TranspositionEntry] = {}
        self._max_size = max_size

    def store(self, key: int, entry: TranspositionEntry) -> None:
        """存储置换表条目。

        Args:
            key: 局面哈希值
            entry: 置换表条目
        """
        # 如果表已满，使用深度优先的替换策略
        if len(self._table) >= self._max_size:
            # 寻找深度最小且非精确值的条目进行替换（优先保留深度大的和精确值）
            lowest_priority_key = None
            lowest_priority = float('inf')
            
            for k, v in self._table.items():
                # 优先级计算：优先移除深度小的，其次是上界，然后是下界，最后是精确值
                priority = -v.depth  # 深度越小优先替换
                if v.flag == TranspositionFlag.UPPER_BOUND:
                    priority -= 1  # 上界略优先于下界
                elif v.flag == TranspositionFlag.LOWER_BOUND:
                    priority -= 0.5
                elif v.flag == TranspositionFlag.EXACT:
                    priority -= 0.1  # 精确值最不优先替换
            
                if priority < lowest_priority:
                    lowest_priority = priority
                    lowest_priority_key = k
            
            if lowest_priority_key is not None:
                del self._table[lowest_priority_key]
            else:
                # 如果没找到合适的替换目标，删除第一个条目
                oldest_key = next(iter(self._table))
                del self._table[oldest_key]

        self._table[key] = entry

    def probe(self, key: int) -> Optional[TranspositionEntry]:
        """查询置换表条目。

        Args:
            key: 局面哈希值

        Returns:
            对应的置换表条目，如果不存在则返回None
        """
        return self._table.get(key)

    def clear(self) -> None:
        """清空置换表。"""
        self._table.clear()


class MoveOrderingStrategy(Protocol):
    """Negamax 展开前的走法排序协议。"""

    def order_moves(
        self,
        board: BoardState,
        moves: Iterable[Move],
        principal_variation: Optional[Sequence[Move]] = None,
    ) -> List[Move]:
        """根据 Alpha-Beta 需求返回排好序的走法列表。"""


@dataclass
class SearchConfig:
    """迭代加深 Negamax 搜索的配置选项。"""

    max_depth: int
    time_limit_seconds: Optional[float] = None
    aspiration_window: Optional[Tuple[float, float]] = None
    use_transposition_table: bool = True
    use_iterative_deepening: bool = True
    use_null_move_pruning: bool = False
    use_quiescence_search: bool = False


@dataclass
class SearchStatistics:
    """搜索过程中的统计指标。"""

    nodes_searched: int = 0
    cutoffs: int = 0
    transposition_hits: int = 0
    max_depth_reached: int = 0
    principal_variation: List[Move] = field(default_factory=list)


@dataclass
class SearchResult:
    """一次搜索调用的结果。"""

    best_move: Optional[Move]
    score: float
    depth: int
    statistics: SearchStatistics


class IterativeDeepeningNegamax:
    """结合 Alpha-Beta 剪枝及多种增强策略的 Negamax 搜索框架。"""

    def __init__(
        self,
        evaluator: HeuristicEvaluator,
        move_orderer: MoveOrderingStrategy,
        transposition_table: Optional[TranspositionTable] = None,
        zobrist_hasher: Optional[ZobristHasher] = None,
        config: Optional[SearchConfig] = None,
    ) -> None:
        self._evaluator = evaluator
        self._move_orderer = move_orderer
        self._transposition_table = transposition_table
        self._zobrist_hasher = zobrist_hasher
        self._config = config or SearchConfig(max_depth=6)

    def search(self, board: BoardState) -> SearchResult:
        """从 ``board`` 出发执行迭代加深 Negamax 搜索。"""
        
        statistics = SearchStatistics()
        best_move = None
        best_score = float("-inf")
        
        # 记录开始时间以便检查超时
        start_time = time.time()

        # 迭代加深循环
        for depth in range(1, self._config.max_depth + 1):
            statistics.max_depth_reached = depth
            
            # 检查时间限制
            if (self._config.time_limit_seconds is not None and 
                time.time() - start_time >= self._config.time_limit_seconds):
                break

            # 执行当前深度的搜索
            try:
                current_score, current_move = self._negamax_with_timeout(
                    board, depth, float("-inf"), float("inf"), statistics, start_time
                )
                
                # 更新最佳走法
                if current_move is not None:
                    best_move = current_move
                    best_score = current_score
                    
                    # 如果找到必胜走法，立即返回
                    if current_score >= 50000:
                        break
                        
            except Exception as e:
                # 如果搜索出错，使用当前最佳走法
                print(f"Search error at depth {depth}: {e}")
                break

        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=statistics.max_depth_reached,
            statistics=statistics,
        )

    def _negamax(
        self,
        board: BoardState,
        depth: int,
        alpha: float,
        beta: float,
        statistics: SearchStatistics,
    ) -> Tuple[float, Optional[Move]]:
        """带 Alpha-Beta 剪枝的 Negamax 递归核心。"""
        # 使用当前时间作为开始时间
        start_time = time.time()
        return self._negamax_with_timeout(board, depth, alpha, beta, statistics, start_time)

    def _negamax_with_timeout(
        self,
        board: BoardState,
        depth: int,
        alpha: float,
        beta: float,
        statistics: SearchStatistics,
        start_time: float,
    ) -> Tuple[float, Optional[Move]]:
        """带 Alpha-Beta 剪枝和超时检查的 Negamax 递归核心。"""

        # 增加节点计数
        statistics.nodes_searched += 1

        # 检查时间限制
        if (self._config.time_limit_seconds is not None and 
            time.time() - start_time >= self._config.time_limit_seconds):
            # 返回当前评估值作为近似结果
            evaluation = self._evaluator.evaluate(board)
            return evaluation.total, None

        # 检查是否为终端节点
        if board.is_terminal() or depth <= 0:
            # 到达叶子节点，使用评估函数
            evaluation = self._evaluator.evaluate(board)
            return evaluation.total, None

        # 查询置换表
        original_alpha = alpha
        tt_entry = None
        if self._transposition_table is not None and self._zobrist_hasher is not None:
            hash_value = board.zobrist_hash()
            tt_entry = self._transposition_table.probe(hash_value)

            # 如果置换表命中且深度足够，使用缓存结果
            if tt_entry is not None and tt_entry.depth >= depth:
                statistics.transposition_hits += 1

                if tt_entry.flag == TranspositionFlag.EXACT:
                    # 精确值，直接返回
                    return tt_entry.value, tt_entry.best_move
                elif tt_entry.flag == TranspositionFlag.LOWER_BOUND:
                    # 下界值，调整alpha
                    alpha = max(alpha, tt_entry.value)
                elif tt_entry.flag == TranspositionFlag.UPPER_BOUND:
                    # 上界值，调整beta
                    beta = min(beta, tt_entry.value)

                # 如果alpha >= beta，提前返回
                if alpha >= beta:
                    return tt_entry.value, tt_entry.best_move

        # 获取当前玩家的合法走法
        legal_moves = list(board.legal_moves())

        # 如果没有合法走法，返回评估值
        if not legal_moves:
            evaluation = self._evaluator.evaluate(board)
            return evaluation.total, None

        # 获取走法排序
        ordered_moves = (
            self._move_orderer.order_moves(board, legal_moves)
            if self._move_orderer is not None
            else list(legal_moves)
        )

        best_move = None
        best_score = float("-inf")

        # 遍历所有走法
        for move in ordered_moves:
            # 创建棋盘副本并执行走法
            board_copy = board.clone()
            board_copy.apply_move(move)

            # 递归搜索
            score, _ = self._negamax_with_timeout(board_copy, depth - 1, -beta, -alpha, statistics, start_time)

            # Negamax 算法：取负值
            score = -score

            # 更新最佳分数和走法
            if score > best_score:
                best_score = score
                best_move = move

            # Alpha-Beta 剪枝
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta 剪枝
                statistics.cutoffs += 1
                break

        # 存储到置换表
        if self._transposition_table is not None and self._zobrist_hasher is not None:
            hash_value = board.zobrist_hash()

            # 确定评值类型
            flag = TranspositionFlag.EXACT
            if best_score <= original_alpha:
                flag = TranspositionFlag.UPPER_BOUND  # 失败低（上界）
            elif best_score >= beta:
                flag = TranspositionFlag.LOWER_BOUND  # 失败高（下界）

            # 创建置换表条目
            entry = TranspositionEntry(
                value=best_score, depth=depth, flag=flag, best_move=best_move
            )

            # 存储到置换表
            self._transposition_table.store(hash_value, entry)

        return best_score, best_move

    def _order_moves(
        self,
        board: BoardState,
        moves: Iterable[Move],
    ) -> List[Move]:
        """调用配置好的走法排序策略。"""

        raise NotImplementedError("走法排序策略尚未接入。")

    def _probe_transposition(
        self, board: BoardState, depth: int, alpha: float, beta: float
    ) -> Optional[Tuple[float, Optional[Move]]]:
        """尝试从置换表读取缓存结果。"""

        raise NotImplementedError("置换表查询尚未接入。")

    def _store_transposition(
        self,
        board: BoardState,
        depth: int,
        value: float,
        flag: TranspositionFlag,
        best_move: Optional[Move],
    ) -> None:
        """将搜索结果写入置换表。"""

        raise NotImplementedError("置换表写入尚未接入。")

    def _quiescence_search(
        self,
        board: BoardState,
        alpha: float,
        beta: float,
        statistics: SearchStatistics,
    ) -> float:
        """在不稳定局面中延伸搜索以缓解地平线效应。"""

        raise NotImplementedError("静态搜索尚未实现。")

    def _apply_move(
        self,
        board: BoardState,
        move: Move,
        eval_state: EvaluationBreakdown,
    ) -> EvaluationBreakdown:
        """在执行走法时结合增量评估更新。"""

        raise NotImplementedError("走法执行时的增量评估尚未实现。")

    def _revert_move(
        self,
        board: BoardState,
        move: Move,
        eval_state: EvaluationBreakdown,
    ) -> EvaluationBreakdown:
        """撤销走法并恢复增量评估状态。"""

        raise NotImplementedError("走法撤销时的增量评估尚未实现。")


BoardMatrix = (
    np.ndarray
)  # 2D numpy array for chessboard, e.g., np.array([[0,1,0], [1,0,0]])
BoardAdapter = Callable[[BoardMatrix, int], BoardState]
MoveTranslator = Callable[[Move], Tuple[int, int]]


class SimpleMoveOrderer(MoveOrderingStrategy):

    def __init__(self):
        self.evaluator = SimpleEvaluator()

    def _detect_threats(self, board: BoardState, player: int) -> List[Tuple[int, int]]:
        """检测指定玩家的威胁位置（如活三、活四等）"""
        threats = []
        opponent = 3 - player

        # 简单检测对手的活三和活四威胁
        for r in range(board.board_size):
            for c in range(board.board_size):
                if board.board[r, c] == 0:  # 空位
                    # 试着下在这个位置
                    test_board = board.clone()
                    test_move = Move(coordinate=(r, c))
                    test_board.apply_move(test_move)

                    # 检查是否形成关键棋形
                    eval_result = self.evaluator.evaluate(test_board)
                    if test_board.current_player_id == player:
                        # 检查是否阻止了对手的威胁或形成了己方威胁
                        if eval_result.pattern_scores.get(PatternType.LIVE_FOUR, 0) > 0 or \
                           eval_result.pattern_scores.get(PatternType.OPEN_THREE, 0) > 0:
                            threats.append((r, c))

        return threats

    def _quick_evaluate_move(self, board: GomokuBoardState, move: Move) -> float:
        """快速评估走法，不需实际执行走法"""
        r, c = move.coordinate
        
        # 基础分数
        score = 0.0
        current_player = board.current_player_id
        
        # 检查四个方向的连续棋子数
        directions = [
            ("horizontal", (0, 1)),      
            ("vertical", (1, 0)),        
            ("diagonal_main", (1, 1)),   
            ("diagonal_anti", (1, -1))   
        ]
        
        # 快速评估：模拟落子后的连续数
        for direction_name, (dx, dy) in directions:
            # 计算在这个位置落子后可能形成的连续数
            consecutive = self._count_potential_consecutive(board, r, c, dx, dy, current_player)
            
            # 根据连续数评分
            if consecutive >= 5:
                score += 100000.0  # 连五
            elif consecutive == 4:
                # 检查是否为活四
                if self._is_open_pattern(board, r, c, dx, dy, 4):
                    score += 50000.0  # 活四
                else:
                    score += 10000.0  # 冲四
            elif consecutive == 3:
                if self._is_open_pattern(board, r, c, dx, dy, 3):
                    score += 5000.0   # 活三
                else:
                    score += 1000.0   # 眠三
            elif consecutive == 2:
                score += 500.0      # 活二
            else:
                score += consecutive * 10.0
        
        # 中心性奖励
        center = board.board_size / 2.0
        center_distance = abs(r - center) + abs(c - center)
        score -= center_distance * 5.0
        
        return score

    def _count_potential_consecutive(self, board: GomokuBoardState, row: int, col: int, 
                                   dx: int, dy: int, player: int) -> int:
        """计算在指定位置落子后可能形成的连续棋子数。"""
        count = 1  # 包括即将落下的棋子
        
        # 向正方向计数
        x, y = row + dx, col + dy
        while (0 <= x < board.board_size and 0 <= y < board.board_size and 
               board.board[x, y] == player):
            count += 1
            x += dx
            y += dy
            
        # 向负方向计数
        x, y = row - dx, col - dy
        while (0 <= x < board.board_size and 0 <= y < board.board_size and 
               board.board[x, y] == player):
            count += 1
            x -= dx
            y -= dy
            
        return count
    
    def _is_open_pattern(self, board: GomokuBoardState, row: int, col: int,
                        dx: int, dy: int, length: int) -> bool:
        """检查指定方向的棋形是否为开放形态（两端都有空位）。"""
        # 检查正方向末端
        x, y = row + dx * length, col + dy * length
        pos_end_open = (0 <= x < board.board_size and 0 <= y < board.board_size and 
                       board.board[x, y] == 0)
        
        # 检查负方向末端
        x, y = row - dx * length, col - dy * length
        neg_end_open = (0 <= x < board.board_size and 0 <= y < board.board_size and 
                       board.board[x, y] == 0)
        
        return pos_end_open and neg_end_open

    def order_moves(
        self,
        board: BoardState,
        moves: Iterable[Move],
        principal_variation: Optional[Sequence[Move]] = None,
    ) -> List[Move]:
        scored_moves: List[Tuple[float, Move]] = []
        pv_coordinates = {move.coordinate for move in (principal_variation or [])}

        for move in moves:
            try:
                score = 0.0

                # PV move 最高优先级
                if move.coordinate in pv_coordinates:
                    score += 1000000.0

                # 快速评估走法价值
                if isinstance(board, GomokuBoardState):
                    score += self._quick_evaluate_move(board, move)
                else:
                    # 对非GomokuBoardState的后备处理
                    try:
                        board_copy = board.clone()
                        board_copy.apply_move(move)
                        eval_result = self.evaluator.evaluate(board_copy)
                        score += eval_result.total
                    except Exception:
                        # 如果评估失败，使用基础中心性评分
                        r, c = move.coordinate[:2]
                        board_size = getattr(board, 'board_size', 15)
                        center = board_size / 2.0
                        center_distance = abs(r - center) + abs(c - center)
                        score = 100.0 - center_distance

                scored_moves.append((score, move))
                
            except Exception as e:
                # 如果移动评估失败，给予基础分数
                print(f"Move evaluation error: {e}")
                scored_moves.append((1.0, move))

        # 按分数降序排序
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves]


class SimpleEvaluator(IncrementalEvaluator, HeuristicEvaluator):
    """组合现有增量评估器与 HeuristicEvaluator 协议。"""


def default_board_adapter(board: BoardMatrix, player: int) -> GomokuBoardState:
    state = GomokuBoardState(
        board=np.array(board, copy=True),
        current_player_id=player,
        board_size=board.shape[0],
    )
    return state


def default_move_translator(move: Move) -> Tuple[int, int]:
    if not move.coordinate:
        raise ValueError("Move has no coordinate")
    return tuple(int(idx) for idx in move.coordinate[:2])


class Search(Agent):
    """继承自基础 Agent 的搜索代理，整合 Negamax 搜索框架。"""

    def __init__(
        self,
        player: int,
        board_adapter: Optional[BoardAdapter] = None,
        move_translator: Optional[MoveTranslator] = None,
        evaluator: Optional[HeuristicEvaluator] = None,
        move_orderer: Optional[MoveOrderingStrategy] = None,
        transposition_table: Optional[TranspositionTable] = None,
        zobrist_hasher: Optional[ZobristHasher] = None,
        config: Optional[SearchConfig] = None,
    ) -> None:
        super().__init__(player)

        self._board_adapter = board_adapter or default_board_adapter
        self._move_translator = move_translator or default_move_translator
        self._search = IterativeDeepeningNegamax(
            evaluator=evaluator or SimpleEvaluator(),
            move_orderer=move_orderer or SimpleMoveOrderer(),
            transposition_table=transposition_table,
            zobrist_hasher=zobrist_hasher,
            config=config or SearchConfig(max_depth=6, time_limit_seconds=3.0),  # Balanced depth and time limit
        )

    def make_move(self, board: BoardMatrix) -> Optional[Tuple[int, int]]:
        board_state = self._board_adapter(board, self.player)
        search_result = self._search.search(board_state)
        if search_result.best_move is None:
            return None
        return self._move_translator(search_result.best_move)
