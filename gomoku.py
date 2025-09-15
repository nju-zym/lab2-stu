import argparse
import importlib
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

PLAYER_TIME_LIMIT = 60.0


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


def play_game(agent1=None, agent2=None, board_size=15):
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

    print("游戏开始! ")
    print(f"玩家操作时间限制: {PLAYER_TIME_LIMIT}秒")
    print_board(board)

    while not game_over:
        print(f"\n轮到玩家 {current_player} (Agent {current_player})")

        current_agent = agents[current_player]

        start_time = time.time()

        is_human_player = hasattr(current_agent, "create_gui")

        if is_human_player:
            try:
                move = current_agent.make_move(board.copy())
                end_time = time.time()
                print(f"玩家 {current_player} 落子时间: {end_time - start_time:.4f}秒")
            except Exception as e:
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

                    print(
                        f"玩家 {current_player} 落子时间: {end_time - start_time:.4f}秒"
                    )

                except FutureTimeoutError:
                    end_time = time.time()
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
                    print(f"玩家 {current_player} 出现异常: {e}")
                    winner = 3 - current_player
                    game_over = True
                    break

        if game_over:
            break

        if move is None:
            print("Agent无法做出有效移动! ")
            winner = 3 - current_player
            break

        row, col = move

        if not is_valid_move(board, row, col):
            winner = 3 - current_player
            print(f"无效的移动: ({row}, {col}), 对手(Agent {winner})获胜! ")
            break

        make_move(board, row, col, current_player)
        print(f"玩家 {current_player} 在 ({row}, {col}) 落子")
        print_board(board)

        if check_win(board, row, col):
            game_over = True
            winner = current_player
            print(f"玩家 {current_player} 获胜! ")
        elif is_board_full(board):
            game_over = True
            winner = 0
            print("游戏平局! ")
        else:
            current_player = 3 - current_player

    if winner == 0:
        print("\n游戏结果: 平局! ")
    elif winner:
        print(f"\n游戏结果: 玩家 {winner} 获胜! ")

    return winner


def main():
    """主函数, 演示游戏使用"""
    parser = argparse.ArgumentParser(description="五子棋对战")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="random",
        help="A2算法选择: human(人类) 或 xxx(算法模块名)",
    )
    parser.add_argument("-s", "--size", type=int, default=11, help="棋盘大小")
    args = parser.parse_args()

    board_size = args.size
    print(f"创建 {board_size}x{board_size} 的棋盘")

    mod = importlib.import_module("241880200")  # 加载学号同名模块（不带 .py）
    A1 = mod.Search                              # 取出 Search 类作为先手的类

    agent1 = A1(1)

    if args.method == "human":
        from human import Human

        agent2 = Human(2)
    elif args.method == "random":
        from agent import Agent

        agent2 = Agent(2)

    elif args.method == "ai":
        from ai import Search

        agent2 = Search(2)
    else:
        try:
            mod = importlib.import_module(f"{args.method}")
            agent2 = mod.Search(2)
        except Exception as e:
            print(f"无法加载gomoku/{args.method}.py 的Search类: {e}")
            print("请确认该文件存在且有Search类")
            return

    play_game(agent1, agent2, board_size)


if __name__ == "__main__":
    main()
