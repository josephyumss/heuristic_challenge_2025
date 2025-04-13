from pathlib import Path
from random import choice
from time import time
from traceback import print_exc, format_exc
from typing import Tuple, List, Callable, Dict, Generator, Optional, Literal, Union

from action import *
from board import GameBoard
from copy import deepcopy

from queue import PriorityQueue

import logging
class Agent:  # Do not change the name of this class!
    """
    An agent class
    """
    # Do not modify this.
    name = Path(__file__).stem
    _logger = logging.getLogger('SA agent')
    # Do not change the constructor argument!
    def __init__(self, player: Literal['white', 'black']):
        """
        Initialize the agent

        :param player: Player label for this agent. White or Black
        """
        self.player = player

    def heuristic_search(self, board: GameBoard) -> List[Action]:
        goal_row = 8 if self.player=='white' else 0
        current_state=board.get_state()
        current_position = tuple(current_state['player'][self.player]['pawn'])
        frontier = PriorityQueue() 
        frontier.put((8,(current_position, [])))
        reached = {current_position:8}
        initial_pos = current_position

        def evalf(board : GameBoard, initial_pos, cur_pos, next_pos, goal_row, local_path=[0]):
            if len(local_path)==0 : return  board.get_move_turns(cur_pos,next_pos) + abs(next_pos[0]-goal_row)
            sum = board.get_move_turns(initial_pos,local_path[0].position)
            if len(local_path)==1 : return sum + board.get_move_turns(cur_pos,next_pos) + abs(next_pos[0]-goal_row)
            for i in range(len(local_path)-1):
                sum += board.get_move_turns(local_path[i].position,local_path[i+1].position)
            return sum + board.get_move_turns(cur_pos,next_pos) + abs(next_pos[0]-goal_row) #g(n) + h(n)


        while frontier:
            
            current_pos, path = frontier.get(block=False)[1]
            if current_pos[0] == goal_row:
                return path

            if current_pos is not initial_pos:
                current_state = board.simulate_action(None,*path,problem_type=1)

            for action in board.get_applicable_moves(self.player):
                fn = evalf(board,initial_pos,current_pos,action,goal_row,path)
                if action in reached and fn >= reached[action]:
                    continue
                frontier.put((fn,(action, path + [MOVE(self.player,action)])))
                reached[action]=fn
        return []
    

    def local_search(self, board: GameBoard, time_limit: float) -> Union[MOVE, List[BLOCK]]:

        current_state=board.get_state()
        current_position = tuple(current_state['player'][self.player]['pawn'])

        if self.player == 'black' : opponent = 'white'
        else : opponent = 'black'

        def obj(player, cur_pos, opponent_pos):
            pos_diff_x = abs(opponent_pos[0]-cur_pos[0])
            pos_diff_y = abs(opponent_pos[1]-cur_pos[1])
            
            if cur_pos[1]==8 : return 999

            if player =='black':
                # 수평 fence
                v_pos_from_current = (cur_pos[0],cur_pos[1]-1)
                block_turn = board.get_move_turns(cur_pos,v_pos_from_current)

            if player =='white':
                # 수평 fence
                v_pos_from_current = (cur_pos[0],cur_pos[1]+1)
                block_turn = board.get_move_turns(v_pos_from_current,cur_pos)

            W_pos = 1
            W_turn = 1
            return W_turn*block_turn - W_pos*pos_diff_y

        childs = board.get_applicable_moves(self.player)
        while True :
            if not childs :
                 return [BLOCK(self.player,current_position,'horizontal'),]
            neighbor = choice(childs)
            childs.remove(neighbor)
            if obj(neighbor) > obj(current_position) : 
                return MOVE(self.player, neighbor)

        opponent_position = tuple(current_state['player'][opponent]['pawn'])        
        childrens = [child for child in board.get_applicable_moves(self.player)]
        currentK = selectK(board,childrens,currentK,opponent_position)

        return MOVE(self.player,(7,0))
    
        # def select_fence_pos(childrens, currentK, opponent_pos):
        #     every_pos = currentK+childrens

        #     max1 = max(every_pos,key=lambda x : obj(self.player,x,opponent_pos))
        #     every_pos.remove(max1)

        #     max2 = max(every_pos,key=lambda x : obj(self.player,x,opponent_pos))
        #     every_pos.remove(max2)

        #     max3 = max(every_pos,key=lambda x : obj(self.player,x,opponent_pos))
        #     every_pos.remove(max3)

        #     return [max1,max2,max3]

    def belief_state_search(self, board: GameBoard, time_limit: float) -> List[Action]:
        """
        * Complete this function to answer the challenge PART III.

        This function uses belief state search for finding the best move for certain amount of time.
        The system calls your algorithm only once. Your algorithm should consider the time limit.

        You can use your heuristic search or local search function, which is previously implemented, to compute required information.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - AND-OR search and its variants
        - Heuristic/Uninformed search algorithms whose state is actually a belief state.
        - Online DFS algorithm, or other online variant of heuristic/uninformed search.
        - LRTA*

        :param board: The game board with initial game setup.
        :param time_limit: The time limit for the search. Datetime.now() should have lower timestamp value than this.
        :return: The next move.
        """
        raise NotImplementedError()

    def adversarial_search(self, board: GameBoard, time_limit: float) -> Action:
        """
        * Complete this function to answer the challenge PART IV.

        This function uses adversarial search to win the game.
        The system calls your algorithm whenever your turn arrives.
        Each time, it provides new position of your pawn and asks your next decision until time limit is reached.

        You can use your search function, which is previously implemented, to compute relevant information.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - Minimax algorithm, H-minimax algorithm, and Expectminimax algorithm
        - RBFS search
        - Alpha-beta search and heuristic version of it.
        - Pure Monte-Carlo search
        - Monte-Carlo Tree Search and its variants
        - Minimax search with belief states
        - Alpha-beta search with belief states

        :param board: The game board with current state.
        :param time_limit: The time limit for the search. Datetime.now() should have lower timestamp value than this.
        :return: The next move.
        """
        raise NotImplementedError()
    

