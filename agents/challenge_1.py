from pathlib import Path
from random import choice
from time import time
from traceback import print_exc, format_exc
from typing import Tuple, List, Callable, Dict, Generator, Optional, Literal, Union

from action import *
from board import GameBoard

#commit test

class Agent:  # Do not change the name of this class!
    """
    An agent class
    """

    # Do not modify this.
    name = Path(__file__).stem

    # Do not change the constructor argument!
    def __init__(self, player: Literal['white', 'black']):
        """
        Initialize the agent

        :param player: Player label for this agent. White or Black
        """
        self.player = player

    def heuristic_search(self, board: GameBoard) -> List[Action]:
        """
        * Complete this function to answer the challenge PART I.

        This function uses heuristic search for finding the best route to the goal line.
        You have to return a list of action, which denotes the shortest actions toward the goal.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - Breadth-first search
        - Depth-first search
        - Uniform-cost search
        - Greedy Best-first search
        - A* search
        - IDA*
        - RBFS
        - SMA*

        :param board: The game board with initial game setup.
        :return: A list of actions.
        """
        #raise NotImplementedError()
        from collections import deque

        start_pos = board.pawns[board.current_player]  # 현재 플레이어 위치
        goal_line = board.get_goal_line(board.current_player)  # 목표 지점
        
        queue = deque([(start_pos, [])])  # (현재 위치, 이동 경로)
        visited = set()
        
        while queue:
            (current_pos, path) = queue.popleft()
            
            # 목표 지점에 도착하면 액션 리스트 반환
            if current_pos[0] in goal_line:
                return path  

            if current_pos in visited:
                continue
            visited.add(current_pos)
            
            # 다음 가능한 이동 추가
            for action in board.get_legal_actions():  
                new_board = board.copy()
                new_board.apply_action(action)

                new_pos = new_board.pawns[board.current_player]  # 이동 후 위치
                if new_pos not in visited:
                    queue.append((new_pos, path + [action]))

        return []

