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
        applicable_fence = board.get_applicable_fences()

        if self.player == 'black' : opponent = 'white'
        else : opponent = 'black'

        def obj(player,applicable_fence, cur_pos, opponent_pos):
            pos_diff_x = abs(opponent_pos[0]-cur_pos[0])
            pos_diff_y = abs(opponent_pos[1]-cur_pos[1])
            hfence_W_pos = 1
            hfence_W_turn = 1
            vfence_W_pos = 1 
            vfence_W_turn = 1

            if cur_pos[0]==8 : h_fence_obj = -999
            if cur_pos[1]==8 : v_fence_obj = -999
            
            if cur_pos[0] != 8 :
                if player =='black':
                    v_pos_from_current = (cur_pos[0]-1,cur_pos[1])
                    block_turn_v = board.get_move_turns(cur_pos,v_pos_from_current)

                if player =='white':
                    v_pos_from_current = (cur_pos[0]+1,cur_pos[1])
                    block_turn_v = board.get_move_turns(v_pos_from_current,cur_pos)
                
                h_fence_obj = hfence_W_turn*block_turn_v - hfence_W_pos*pos_diff_y
            
            if cur_pos[1] != 8 :
                h_pos_from_current = (cur_pos[0],cur_pos[1]+1)
                block_turn_h = board.get_move_turns(h_pos_from_current,cur_pos)

                v_fence_obj = vfence_W_turn*block_turn_h - vfence_W_pos*pos_diff_x

            if (cur_pos,'horizontal') not in applicable_fence : h_fence_obj = -999
            if (cur_pos,'vertical') not in applicable_fence : v_fence_obj = -999

            return ((h_fence_obj,'horizontal'),(v_fence_obj, 'vertical'))

        def Neighbor_is_better_Position(cur_obj, nbr_obj):
            cur_h_obj = cur_obj[0][0]
            cur_v_obj = cur_obj[1][0]
            nbr_h_obj = nbr_obj[0][0]
            nbr_v_obj = nbr_obj[1][0]

            if (nbr_h_obj > cur_h_obj) and (nbr_v_obj > cur_v_obj): return True
            elif (nbr_h_obj < cur_h_obj) and (nbr_v_obj < cur_v_obj) : return False
            elif (nbr_h_obj == cur_h_obj) and (nbr_v_obj == cur_v_obj): return True
            else :
                h_obj_diff = abs(nbr_h_obj - cur_h_obj) 
                v_obj_diff = abs(nbr_v_obj - cur_v_obj)

                if nbr_h_obj == cur_h_obj:
                    if nbr_v_obj > cur_v_obj : return True
                    else : return False

                if nbr_v_obj == cur_v_obj:
                    if nbr_h_obj > cur_h_obj : return True
                    else : return False

                if nbr_h_obj > cur_h_obj:
                    if h_obj_diff >= v_obj_diff : return True
                    else : return False

                if nbr_v_obj > cur_v_obj:
                    if v_obj_diff >= h_obj_diff : return True
                    else : return False
               
        childs = board.get_applicable_moves(self.player)
        opponent_position = tuple(current_state['player'][opponent]['pawn'])
        cur_pos_obj = obj(self.player,applicable_fence,current_position,opponent_position)
        if cur_pos_obj[0] >= cur_pos_obj[1]:
            best_3_pos = [(cur_pos_obj[0][0],current_position,'horizontal')]
        else : best_3_pos = [(cur_pos_obj[1][0],current_position,'vertical')]

        reached = [current_position]

        while True :
            if not childs :
                if len(best_3_pos)<3 : 
                    self._logger.debug(f"len best 3 pos is not enough : {len(best_3_pos)}")

                    return MOVE(self.player,choice(board.get_applicable_moves()))
                
                self._logger.debug(f"best 3 pos is : {best_3_pos}")
                return [BLOCK(self.player,best_3_pos[0][1],best_3_pos[0][2]),
                        BLOCK(self.player,best_3_pos[1][1],best_3_pos[1][2]),
                        BLOCK(self.player,best_3_pos[2][1],best_3_pos[2][2]),]
            
            neighbor = choice(childs)
            neighbor_obj= obj(self.player,applicable_fence,neighbor,opponent_position) # hobj, vobj
            childs.remove(neighbor)

            if ((neighbor,'horizontal') not in applicable_fence) and ((neighbor,'vertical') not in applicable_fence): 
                continue
            
            # 한번 Move 를 반환하고 나면 reached는 초기화 되므로 동일한 obj 값을 가지는 위치를 왔다 갔다 반복함.
            if Neighbor_is_better_Position(cur_pos_obj,neighbor_obj) and neighbor not in reached:
                self._logger.debug(f"cur_pos_obj : {cur_pos_obj}")
                self._logger.debug(f"neighbor_obj : {neighbor_obj}")
                return MOVE(self.player, neighbor)
            
            if best_3_pos[0][2] == 'horizontal':
                if (neighbor[0] == best_3_pos[0][1][0]-1 or neighbor[0] == best_3_pos[0][1][0]+1):
                    if len(best_3_pos)<3 : best_3_pos += [(neighbor_obj[1][0],neighbor,'vertical')]

                    else :
                        if neighbor_obj[1][0] > min(best_3_pos)[0]:
                            best_3_pos.remove(min(best_3_pos))
                            best_3_pos += [(neighbor_obj[1][0],neighbor,'vertical')]
                
            else : 
                if (neighbor[1] == best_3_pos[0][1][1]-1 or neighbor[1] == best_3_pos[0][1][1]+1):
                    if len(best_3_pos) < 3 : best_3_pos += [(neighbor_obj[0][0],neighbor,'horizontal')]

                    else:
                        if neighbor_obj[0][0] > min(best_3_pos)[0]:
                            best_3_pos.remove(min(best_3_pos))
                            best_3_pos += [(neighbor_obj[0][0],neighbor,'horizontal')]

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
    

