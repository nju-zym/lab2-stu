import argparse
import importlib
import itertools
import math
import numpy as np
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PLAYER_TIME_LIMIT = 60.0
DEFAULT_BOARD_SIZE = 11


def create_board(board_size=15):
    """
    创建棋盘

    @param board_size: 棋盘大小, 默认15x15(标准五子棋棋盘)
    @return: 棋盘数组
    """
    return np.zeros((board_size, board_size), dtype=int)


def is_valid_move(board, row, col):
    """
    检查移动是否有效

    @param board: 棋盘
    @param row: 行坐标
    @param col: 列坐标
    @return: 是否有效
    """
    board_size = len(board)
    return 0 <= row < board_size and 0 <= col < board_size and board[row][col] == 0


def make_move(board, row, col, player):
    """
    在指定位置落子

    @param board: 棋盘
    @param row: 行坐标
    @param col: 列坐标
    @param player: 玩家编号 (1或2)
    @return: 是否成功落子
    """
    if not is_valid_move(board, row, col):
        return False

    board[row][col] = player
    return True


def is_board_full(board):
    """
    检查棋盘是否已满

    @param board: 棋盘
    @return: 是否已满
    """
    return np.all(board != 0)


def check_win(board, row, col):
    """
    检查从指定位置是否形成五子连珠

    @param board: 棋盘
    @param row: 最后落子的行坐标
    @param col: 最后落子的列坐标
    @return: 是否获胜
    """
    board_size = len(board)
    player = board[row][col]

    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]

    for dx, dy in directions:
        count = 1

        x, y = row + dx, col + dy
        while 0 <= x < board_size and 0 <= y < board_size and board[x][y] == player:
            count += 1
            x, y = x + dx, y + dy

        x, y = row - dx, col - dy
        while 0 <= x < board_size and 0 <= y < board_size and board[x][y] == player:
            count += 1
            x, y = x - dx, y - dy

        if count >= 5:
            return True

    return False


def print_board(board):
    """
    打印棋盘

    @param board: 棋盘
    """
    board_size = len(board)
    print("  ", end="")
    for j in range(board_size):
        print(f"{j:2}", end="")
    print()

    for i in range(board_size):
        print(f"{i:2}", end="")
        for j in range(board_size):
            if board[i][j] == 0:
                print(" .", end="")
            elif board[i][j] == 1:
                print(" ●", end="")
            else:
                print(" ○", end="")
        print()


@dataclass(frozen=True)
class AgentSpec:
    label: str
    method: str


@dataclass
class AgentStats:
    name: str
    method: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0


@dataclass
class MatchSummary:
    black: str
    white: str
    winner_label: Optional[str]
    winner_id: int
    duration: float


def instantiate_agent(method: str, player: int, allow_human: bool = True):
    method_key = method.strip()
    method_lower = method_key.lower()

    if method_lower == "human":
        if not allow_human:
            raise ValueError("擂台模式不支持Human玩家")
        from human import Human

        return Human(player)

    if method_lower == "random":
        from agent import Agent

        return Agent(player)

    if method_lower == "z241880200":
        from z241880200 import Search

        return Search(player)

    if method_lower == "search_framework":
        from search_framework import Search

        return Search(player)

    if method_lower == "search_framework_simple":
        from search_framework_simple import Search

        return Search(player)

    if method_lower == "test":
        from test import Search

        return Search(player)

    try:
        module = importlib.import_module(method_key)
    except ImportError as exc:
        raise ValueError(f"无法加载模块 {method_key}: {exc}") from exc

    if hasattr(module, "Search"):
        return module.Search(player)

    raise ValueError(f"模块 {method_key} 未提供 Search 类，无法创建Agent")


def create_agent_factory(
    method: str, *, allow_human: bool = False
) -> Callable[[int], object]:
    def factory(player: int):
        return instantiate_agent(method, player, allow_human=allow_human)

    return factory


def execute_match(
    black_label: str,
    white_label: str,
    factories: Dict[str, Callable[[int], object]],
    board_size: int,
    verbose_games: bool,
) -> MatchSummary:
    agent_black = agent_white = None
    forfeited: Optional[Tuple[str, str]] = None

    start_ts = time.time()

    try:
        agent_black = factories[black_label](1)
    except Exception as exc:
        forfeited = (black_label, str(exc))

    if forfeited is None:
        try:
            agent_white = factories[white_label](2)
        except Exception as exc:
            forfeited = (white_label, str(exc))

    if forfeited is not None:
        offender, reason = forfeited
        winner = 2 if offender == black_label else 1
        duration = time.time() - start_ts
        print(f"{offender} 无法就绪: {reason}. 对手判胜。")
        return MatchSummary(
            black=black_label,
            white=white_label,
            winner_label=white_label if winner == 2 else black_label,
            winner_id=winner,
            duration=duration,
        )

    winner = play_game(
        agent_black,
        agent_white,
        board_size=board_size,
        verbose=verbose_games,
    )
    duration = time.time() - start_ts

    if winner == 1:
        winner_label = black_label
    elif winner == 2:
        winner_label = white_label
    else:
        winner_label = None

    return MatchSummary(
        black=black_label,
        white=white_label,
        winner_label=winner_label,
        winner_id=winner,
        duration=duration,
    )


def apply_match_result(stats: Dict[str, AgentStats], summary: MatchSummary):
    stats[summary.black].games += 1
    stats[summary.white].games += 1

    if summary.winner_id == 1:
        stats[summary.black].wins += 1
        stats[summary.white].losses += 1
        stats[summary.black].points += 1.0
    elif summary.winner_id == 2:
        stats[summary.white].wins += 1
        stats[summary.black].losses += 1
        stats[summary.white].points += 1.0
    else:
        stats[summary.black].draws += 1
        stats[summary.white].draws += 1
        stats[summary.black].points += 0.5
        stats[summary.white].points += 0.5


def print_match_summary(summary: MatchSummary):
    if summary.winner_id == 0:
        outcome = "平局"
    elif summary.winner_id == 1:
        outcome = f"{summary.black} (黑) 获胜"
    else:
        outcome = f"{summary.white} (白) 获胜"

    print(
        f"[{summary.black} (黑) vs {summary.white} (白)] -> {outcome}，耗时 {summary.duration:.2f} 秒"
    )


def print_rankings(stats: Dict[str, AgentStats]):
    print("\n=== 擂台排名 ===")
    sorted_stats = sorted(
        stats.values(), key=lambda item: (-item.points, -item.wins, item.name.lower())
    )

    max_name_len = max((len(item.name) for item in sorted_stats), default=4)
    header = (
        f"{'名次':<6}{'AI名称':<{max_name_len + 2}}{'局数':<6}{'胜':<4}"
        f"{'平':<4}{'负':<4}{'积分':<6}"
    )
    print(header)
    print("-" * len(header))

    last_points: Optional[float] = None
    current_rank = 0

    for idx, item in enumerate(sorted_stats, start=1):
        if last_points is None or not math.isclose(item.points, last_points):
            current_rank = idx
            last_points = item.points

        print(
            f"{current_rank:<6}{item.name:<{max_name_len + 2}}{item.games:<6}"
            f"{item.wins:<4}{item.draws:<4}{item.losses:<4}{item.points:<6.1f}"
        )


def parse_agent_specs(raw_agents: Sequence[str]) -> List[AgentSpec]:
    specs: List[AgentSpec] = []

    for raw in raw_agents:
        if not raw:
            continue

        if "=" in raw:
            label, method = raw.split("=", 1)
            label = label.strip()
            method = method.strip()
        else:
            label = raw.strip()
            method = raw.strip()

        if not label or not method:
            raise ValueError(f"非法的AI配置: '{raw}'")

        specs.append(AgentSpec(label=label, method=method))

    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("AI 名称存在重复，请为每个AI提供唯一名称")

    return specs


def run_arena(
    agent_specs: Sequence[AgentSpec],
    board_size: int,
    workers: int = 1,
    verbose: bool = False,
):
    if len(agent_specs) < 2:
        raise ValueError("擂台模式至少需要两个AI")

    if workers < 1:
        workers = 1

    factories: Dict[str, Callable[[int], object]] = {
        spec.label: create_agent_factory(spec.method, allow_human=False)
        for spec in agent_specs
    }

    stats: Dict[str, AgentStats] = {
        spec.label: AgentStats(name=spec.label, method=spec.method)
        for spec in agent_specs
    }

    pairings: List[Tuple[str, str]] = []
    for left, right in itertools.combinations(agent_specs, 2):
        pairings.append((left.label, right.label))
        pairings.append((right.label, left.label))

    total_matches = len(pairings)
    game_verbose = verbose and workers == 1

    print(
        f"启动AI擂台: {len(agent_specs)} 个AI, 总对局数 {total_matches}, "
        f"棋盘大小 {board_size}x{board_size}, 并行线程 {workers}"
    )

    match_summaries: List[MatchSummary] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                execute_match, black, white, factories, board_size, game_verbose
            ): (black, white)
            for black, white in pairings
        }

        for future in as_completed(future_map):
            try:
                summary = future.result()
            except Exception as exc:
                black, white = future_map[future]
                print(f"对局 {black} vs {white} 发生异常: {exc}")
                continue
            apply_match_result(stats, summary)
            match_summaries.append(summary)
            print_match_summary(summary)

    print("\n全部对局结束！")
    print_rankings(stats)

    return match_summaries, stats


def play_game(agent1=None, agent2=None, board_size=15, verbose=True):
    """
    进行一局游戏

    @param agent1: 玩家1的Agent, 如果为None则使用默认Agent
    @param agent2: 玩家2的Agent, 如果为None则使用默认Agent
    @param board_size: 棋盘大小, 默认15x15(标准五子棋棋盘)
    """
    board = create_board(board_size)
    current_player = 1
    game_over = False
    winner = None

    agents = {1: agent1, 2: agent2}

    if verbose:
        print("游戏开始! ")
        print(f"玩家操作时间限制: {PLAYER_TIME_LIMIT}秒")
        print_board(board)

    while not game_over:
        if verbose:
            print(f"\n轮到玩家 {current_player} (Agent {current_player})")

        current_agent = agents[current_player]

        start_time = time.time()

        is_human_player = hasattr(current_agent, "create_gui")

        if is_human_player:
            try:
                move = current_agent.make_move(board.copy())
                end_time = time.time()
                if verbose:
                    print(
                        f"玩家 {current_player} 落子时间: {end_time - start_time:.4f}秒"
                    )
            except Exception as e:
                if verbose:
                    print(f"玩家 {current_player} 出现异常: {e}")
                winner = 3 - current_player
                game_over = True
                break
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(current_agent.make_move, board.copy())
                    move = future.result(timeout=PLAYER_TIME_LIMIT)
                    end_time = time.time()

                    if verbose:
                        print(
                            f"玩家 {current_player} 落子时间: {end_time - start_time:.4f}秒"
                        )

                except FutureTimeoutError:
                    end_time = time.time()
                    if verbose:
                        print(
                            f"玩家 {current_player} 操作超时! 超时时间: {end_time - start_time:.4f}秒"
                        )
                        print(
                            f"超过了 {PLAYER_TIME_LIMIT}秒的时间限制，玩家 {current_player} 败北!"
                        )
                    winner = 3 - current_player
                    game_over = True
                    break
                except Exception as e:
                    if verbose:
                        print(f"玩家 {current_player} 出现异常: {e}")
                    winner = 3 - current_player
                    game_over = True
                    break

        if game_over:
            break

        if move is None:
            if verbose:
                print("Agent无法做出有效移动! ")
            winner = 3 - current_player
            break

        row, col = move

        if not is_valid_move(board, row, col):
            winner = 3 - current_player
            if verbose:
                print(f"无效的移动: ({row}, {col}), 对手(Agent {winner})获胜! ")
            break

        make_move(board, row, col, current_player)
        if verbose:
            print(f"玩家 {current_player} 在 ({row}, {col}) 落子")
            print_board(board)

        if check_win(board, row, col):
            game_over = True
            winner = current_player
            if verbose:
                print(f"玩家 {current_player} 获胜! ")
        elif is_board_full(board):
            game_over = True
            winner = 0
            if verbose:
                print("游戏平局! ")
        else:
            current_player = 3 - current_player

    if verbose:
        if winner == 0:
            print("\n游戏结果: 平局! ")
        elif winner:
            print(f"\n游戏结果: 玩家 {winner} 获胜! ")

    return winner


def main():
    parser = argparse.ArgumentParser(description="五子棋AI擂台")
    parser.add_argument(
        "--arena",
        action="store_true",
        help="启用擂台模式，对多个AI进行循环对战",
    )
    parser.add_argument(
        "-a",
        "--agents",
        nargs="+",
        default=[],
        help="擂台模式下的AI列表，格式 name=module (若省略 name 则使用模块名)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="擂台模式并发线程数",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="random",
        help="单局模式下的对手模块，human/random/search_framework/...",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=DEFAULT_BOARD_SIZE,
        help="棋盘大小",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出每一步详情（仅单线程游戏时有效）",
    )
    args = parser.parse_args()

    board_size = args.size

    if args.arena:
        try:
            specs = parse_agent_specs(args.agents)
        except ValueError as exc:
            print(f"AI配置错误: {exc}")
            return

        try:
            run_arena(
                specs, board_size=board_size, workers=args.workers, verbose=args.verbose
            )
        except ValueError as exc:
            print(f"擂台启动失败: {exc}")
    else:
        print(f"创建 {board_size}x{board_size} 的棋盘")
        agent1 = instantiate_agent("search_framework_simple", 1)
        try:
            agent2 = instantiate_agent(args.method, 2)
        except Exception as exc:
            print(f"无法创建对手AI: {exc}")
            return
        play_game(agent1, agent2, board_size, verbose=True)


if __name__ == "__main__":
    main()
