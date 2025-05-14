from pathlib import Path
from random import choice,randint,sample,shuffle
from time import time
from traceback import print_exc, format_exc
from typing import Tuple, List, Callable, Dict, Generator, Optional, Literal, Union

from action import *
from board import GameBoard
from copy import deepcopy

from queue import PriorityQueue

import logging
import copy
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
    

    def local_search(self, board: GameBoard, time_limit: float) -> List[BLOCK]:
        current_state=board.get_state()
        current_position = tuple(current_state['player'][self.player]['pawn'])

        self._logger.debug(f"current state : {current_state}")

        if self.player == 'black' : opponent = 'white'
        else : opponent = 'black'

        #left fence
        left_fence = board.number_of_fences_left(self.player)
        self._logger.debug(f"left_fence is {left_fence}")

        def check_valid(pos,ori,FC,FH,FV):
            if pos in FC: 
                return False
            
            if ori == "horizontal":
                for fence in FH:
                    if fence[0] == pos[0] :
                        if abs(pos[1]-fence[1])==1:
                            return False
                    
                        if abs(pos[1]-fence[1])==2:
                            if pos[1]==0 or pos[1]==7:
                                return False
                            
                            if [pos[0],pos[1]+1] in FV or [pos[0],pos[1]-1] in FV:
                                return False
                            if [pos[0]+1,pos[1]+1] in FV or [pos[0]+1,pos[1]-1] in FV:
                                return False
                            if [pos[0]-1,pos[1]+1] in FV or [pos[0]-1,pos[1]-1] in FV:
                                return False
                            if [pos[0]-1,pos[1]] in FV or [pos[0]+1,pos[1]] in FV:
                                return False
                            
                            if pos[1] > fence[1]:
                                if [pos[0],pos[1]+2] in FH:
                                    return False
                            if pos[1] < fence[1]:
                                if [pos[0],pos[1]-2] in FH:
                                    return False
                for fence in FV:
                    if pos[1]==fence[1]+1 or pos[1]==fence[1]-1:
                        if pos[1]==0 or pos[1]==7:
                            return False
                        
                    if pos[1]==fence[1]+1 and abs(pos[0]-fence[0]) <= 1:
                        if [pos[0]+1,pos[1]+1] in FV or [pos[0],pos[1]+1] in FV or [pos[0]-1,pos[1]+1] in FV:
                            return False     
                    
                    if pos[1]==fence[1]-1 and abs(pos[0]-fence[0]) <= 1:
                        if [pos[0]+1,pos[1]-1] in FV or [pos[0],pos[1]-1] in FV or [pos[0]-1,pos[1]-1] in FV:
                            return False
            if ori == 'vertical':
                return False
            return True

        # occupied fences
        fenceCenter = current_state['board']['fence_center']
        fenceHorizontal = current_state['board']['horizontal_fences']
        fenceVertical = current_state['board']['vertical_fences']

        #initial fence position
        fence_position = board.get_applicable_fences(self.player)
        candidate = []
        while left_fence > 0 :
            self._logger.debug(f"left_fence is {left_fence}")
            fence = fence_position[0]
            if check_valid(fence[0],fence[1],fenceCenter,fenceHorizontal,fenceVertical):
                candidate.append(fence)
                fence_position.remove(fence)
                left_fence -= 1
                fenceCenter.append(fence[0])
                if fence[1]=='horizontal':
                    fenceHorizontal.append(fence[0])
                else :
                    fenceVertical.append(fence[0])
            else :
                fence_position.remove(fence)
        self._logger.debug(f"the result is {candidate}")

        def candidate_to_BLOCK(candidate : list):
            return [BLOCK(self.player,fence[0],fence[1]) for fence in candidate]

        def generate_neighbor(candidate,applicable_fence,FC,FH,FV):
            copied_candidate = deepcopy(candidate)
            applicable_fence = [fence for fence in applicable_fence if (fence not in copied_candidate) and (check_valid(fence[0],fence[1],FC,FH,FV))]
            n = randint(1,5)
            for i in range(n):
                copied_candidate.pop()
            new_cand = sample(applicable_fence,n)
            new_cand = copied_candidate + new_cand
            return new_cand

        def obj(state, fence, opponent):
            self._logger.debug("compute obj..")
            state = board.simulate_action(None,*candidate_to_BLOCK(fence))
            self._logger.debug("simulate done")
            return board.distance_to_goal(opponent,state)
        
        def change_to_neighbor(cur_cand, neighbor_cand, FC, FH, FV):
            for fence in cur_cand:
                if list(fence[0]) in FC:
                    if fence[1]=="horizontal":
                        FC.remove(list(fence[0]))
                        FH.remove(list(fence[0]))
                    if fence[1]=="vertical":
                        FC.remove(list(fence[0]))
                        FV.remove(list(fence[0]))

            for fence in neighbor_cand:
                if list(fence[0]) not in FC:
                    if fence[1]=="horizontal":
                        FC.append(list(fence[0]))
                        FH.append(list(fence[0]))
                    if fence[1]=="vertical":
                        FC.append(list(fence[0]))
                        FV.append(list(fence[0]))

            return neighbor_cand, FC, FH, FV
        
        self._logger.debug(F"Candidate : {candidate}")
        self._logger.debug(f"FC : {fenceCenter}")
        self._logger.debug(f"FH : {fenceHorizontal}")
        self._logger.debug(f"FV: {fenceVertical}")
        
        count = 0
        while True:
            shuffle(candidate)
            neighbor = generate_neighbor(candidate,board.get_applicable_fences(),fenceCenter,fenceHorizontal,fenceVertical) # FC,FH,FV가 update되어야 함. 여기 수정.
            self._logger.debug(f"neighbor is {neighbor}")
            current_obj = obj(current_state, candidate, opponent)
            self._logger.debug(f"cur obj : {current_obj}")
            neighbor_obj = obj(current_state, neighbor, opponent)
            self._logger.debug(f"cur obj : {current_obj}, neighbor obj : {neighbor_obj}")
            if neighbor_obj > current_obj: # 일단은 side walk 불가능 조건으로 둠. 이후 reached 만들어서 side walk 만들어도 될 듯
                self._logger.debug(f"change to neighbor : {candidate} -> {neighbor}")
                candidate, fenceCenter, fenceHorizontal, fenceVertical = change_to_neighbor(candidate,neighbor,fenceCenter, fenceHorizontal, fenceVertical)
            elif count==10:
                break
            else :
                count += 1
            
        return [BLOCK(self.player,fence[0],fence[1]) for fence in candidate]
                    
        # self._logger.debug("get applicable moves for childs")
        # childs = board.get_applicable_moves(self.player)
        # opponent_position = tuple(current_state['player'][opponent]['pawn'])
        # cur_pos_obj = obj(self.player,applicable_fence,current_position,opponent_position)
        # if cur_pos_obj[0] >= cur_pos_obj[1]:
        #     best_3_pos = [(cur_pos_obj[0][0],current_position,'horizontal')]
        # else : best_3_pos = [(cur_pos_obj[1][0],current_position,'vertical')]

        # reached = [current_position]

        # while True :
        #     if not childs :
        #         if len(best_3_pos)<3 : 
        #             self._logger.debug(f"len best 3 pos is not enough : {len(best_3_pos)}")
        #             current_state = board.simulate_action(None,MOVE(self.player,choice(board.get_applicable_moves())),problem_type=2)
        #             break

        #         self._logger.debug(f"best 3 pos is : {best_3_pos}")
        #         return [BLOCK(self.player,best_3_pos[0][1],best_3_pos[0][2]),
        #                 BLOCK(self.player,best_3_pos[1][1],best_3_pos[1][2]),
        #                 BLOCK(self.player,best_3_pos[2][1],best_3_pos[2][2]),]
            
        #     neighbor = choice(childs)
        #     neighbor_obj= obj(self.player,applicable_fence,neighbor,opponent_position) # hobj, vobj
        #     childs.remove(neighbor)

        #     if ((neighbor,'horizontal') not in applicable_fence) and ((neighbor,'vertical') not in applicable_fence): 
        #         continue
            
        #     # 한번 Move 를 반환하고 나면 reached는 초기화 되므로 동일한 obj 값을 가지는 위치를 왔다 갔다 반복함.
        #     if Neighbor_is_better_Position(cur_pos_obj,neighbor_obj) and neighbor not in reached:
        #         self._logger.debug(f"cur_pos_obj : {cur_pos_obj}")
        #         self._logger.debug(f"neighbor_obj : {neighbor_obj}")
        #         try:
        #             self._logger.debug(f"before simulation, left fence is {board.number_of_fences_left(self.player)}")
        #             current_state = board.simulate_action(None,MOVE(self.player,neighbor),problem_type=2)
        #             self._logger.debug(f"after simulation, left fence is {board.number_of_fences_left(self.player)}")
        #         except :
        #             self._logger.debug(f"opponent STUCK")  
            
        #     if best_3_pos[0][2] == 'horizontal':
        #         if (neighbor[0] == best_3_pos[0][1][0]-1 or neighbor[0] == best_3_pos[0][1][0]+1):
        #             if len(best_3_pos)<3 : best_3_pos += [(neighbor_obj[1][0],neighbor,'vertical')]

        #             else :
        #                 if neighbor_obj[1][0] > min(best_3_pos)[0]:
        #                     best_3_pos.remove(min(best_3_pos))
        #                     best_3_pos += [(neighbor_obj[1][0],neighbor,'vertical')]
                
        #     else : 
        #         if (neighbor[1] == best_3_pos[0][1][1]-1 or neighbor[1] == best_3_pos[0][1][1]+1):
        #             if len(best_3_pos) < 3 : best_3_pos += [(neighbor_obj[0][0],neighbor,'horizontal')]

        #             else:
        #                 if neighbor_obj[0][0] > min(best_3_pos)[0]:
        #                     best_3_pos.remove(min(best_3_pos))
        #                     best_3_pos += [(neighbor_obj[0][0],neighbor,'horizontal')]

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
    

