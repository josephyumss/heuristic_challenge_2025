from pathlib import Path
from random import choice
from time import time
from traceback import print_exc, format_exc
from typing import Tuple, List, Callable, Dict, Generator, Optional, Literal, Union

from action import *
from board import GameBoard

#commit test
import logging
import sys
IS_DEBUG = '--debug' in sys.argv
from copy import deepcopy

class Agent:  # Do not change the name of this class!
    """
    An agent class
    """
    #여기
    _logger = logging.getLogger('agent')    
    _logger.propagate = True

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
     
        from collections import deque

        #현재 agent의 위치?
        current_state=board.get_state()
        current_position = tuple(current_state['player'][self.player]['pawn']) #tuple로 위치 나옴
        goal_row = 8 if self.player=='white' else 0

        queue = deque([(current_position, [])]) 
        visited = set()
        
        while queue:

            current_pos, path = queue.popleft()

            if current_pos[0] == goal_row:
                self._logger.debug(f'tmp state : {tmp_state}')
                return path  

            if current_pos in visited:
                continue

            visited.add(current_pos)

            tmp_state=deepcopy(current_state)
            tmp_state['player'][self.player]['pawn']=[current_pos[0],current_pos[1]]   # 실행 하려면! 여기 지우고

            tmp_state = board.simulate_action(tmp_state,*[],problem_type=1)   # 실행 하려면! 여기서 tmp_state 대신 None

            for action in board.get_applicable_moves(self.player):   #action은 tuple (row,col)
                self._logger.debug(f'applicable moves : {action}')  # 왜 벽을 건너뛰는 move가 가능하다고 나오지?
                if action not in visited:
                    queue.append((action, path + [MOVE(self.player,action)]))

        return []

