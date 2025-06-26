from pathlib import Path
from random import choice,randint,sample,shuffle
from time import time
from traceback import print_exc, format_exc
from typing import Tuple, List, Callable, Dict, Generator, Optional, Literal, Union

from action import *
from board import GameBoard
from copy import deepcopy

from queue import PriorityQueue
import math
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
        # board.simulate_action(None,*[BLOCK(self.player,(4,3),'horizontal'),BLOCK(self.player,(6,7),'horizontal'),BLOCK(self.player,(2,7),'horizontal'),
        #                              BLOCK(self.player,(3,7),'horizontal'),BLOCK(self.player,(6,4),'horizontal'),BLOCK(self.player,(5,1),'horizontal')])
        # return None
        current_state=board.get_state()
        current_position = tuple(current_state['player'][self.player]['pawn'])
        
        self._logger.debug(f"current state : {current_state}")

        if self.player == 'black' : opponent = 'white'
        else : opponent = 'black'
        self._logger.debug(f"board.distance_to_goal(opponent) : {board.distance_to_goal(opponent)}")
        opponent_position = tuple(current_state['player'][opponent]['pawn'])

        #left fence
        left_fence = board.number_of_fences_left(self.player)
        self._logger.debug(f"left_fence is {left_fence}")

        def check_valid(pos,ori,FC,FH,FV):
            if list(pos) in FC: 
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
                    if abs(pos[0]-fence[0]) <= 1:
                        if pos[1]==fence[1]+1 or pos[1]==fence[1]-1 or pos[1]==fence[1]:
                            if pos[1]==0 or pos[1]==7:
                                return False
                        
                        if pos[1]==fence[1]+1:
                            if [pos[0]+1,pos[1]+1] in FV or [pos[0],pos[1]+1] in FV or [pos[0]-1,pos[1]+1] in FV or [pos[0]+1,pos[1]] in FV or [pos[0]-1,pos[1]] in FV:
                                return False     

                        if pos[1]==fence[1]-1:
                            if [pos[0]+1,pos[1]-1] in FV or [pos[0],pos[1]-1] in FV or [pos[0]-1,pos[1]-1] in FV or [pos[0]+1,pos[1]] in FV or [pos[0]-1,pos[1]] in FV:
                                return False
            if ori == 'vertical':
                for fence in FV:
                    if fence[1] == pos[1] :
                        if abs(pos[0]-fence[0])==1:
                            return False
                    
                        if abs(pos[0]-fence[0])==2:
                            if pos[0]==0 or pos[0]==7:
                                return False
                            
                            if [pos[0]+1,pos[1]] in FH or [pos[0]-1,pos[1]] in FH:
                                return False
                            if [pos[0]+1,pos[1]+1] in FH or [pos[0]+1,pos[1]-1] in FH:
                                return False
                            if [pos[0]-1,pos[1]+1] in FH or [pos[0]-1,pos[1]-1] in FH:
                                return False
                            if [pos[0],pos[1]+1] in FH or [pos[0],pos[1]-1] in FH:
                                return False
                            
                            if pos[0] > fence[0]:
                                if [pos[0]+2,pos[1]] in FV:
                                    return False
                            if pos[1] < fence[1]:
                                if [pos[0]-2,pos[1]] in FV:
                                    return False
                for fence in FH:
                    if abs(pos[1]-fence[1]) <= 1:
                        if pos[0]==fence[0]+1 or pos[0]==fence[0]-1 or pos[0]==fence[0]:
                            if pos[0]==0 or pos[0]==7:
                                return False
                            
                        if pos[0]==fence[0]+1:
                            if [pos[0]+1,pos[1]+1] in FH or [pos[0]+1,pos[1]] in FH or [pos[0]+1,pos[1]-1] in FH:
                                return False     
                        
                        if pos[0]==fence[0]-1:
                            if [pos[0]-1,pos[1]+1] in FH or [pos[0]-1,pos[1]] in FH or [pos[0]-1,pos[1]-1] in FH:
                                return False
            return True

        # occupied fences
        fenceCenter = current_state['board']['fence_center']
        fenceHorizontal = current_state['board']['horizontal_fences']
        fenceVertical = current_state['board']['vertical_fences']
        self._logger.debug(f"FC : {fenceCenter}")
        self._logger.debug(f"FH : {fenceHorizontal}")
        self._logger.debug(f"FV : {fenceVertical}")
        
        #initial fence position
        fence_position = board.get_applicable_fences(self.player)
        candidate = []
        while left_fence > 0 :
            self._logger.debug(f"left_fence is {left_fence}")
            if opponent == 'white':
                fence = fence_position[0]
            if opponent == 'black':
                fence = fence_position[-1]
            if check_valid(fence[0],fence[1],fenceCenter,fenceHorizontal,fenceVertical):
                candidate.append(fence)
                fence_position.remove(fence)
                left_fence -= 1
                fenceCenter.append(list(fence[0]))
                if fence[1]=='horizontal':
                    fenceHorizontal.append(list(fence[0]))
                else :
                    fenceVertical.append(list(fence[0]))
            else :
                fence_position.remove(fence)
        self._logger.debug(f"the result is {candidate}")
        self._logger.debug(f"FC : {fenceCenter}")
        self._logger.debug(f"FH : {fenceHorizontal}")
        self._logger.debug(f"FV : {fenceVertical}")
        

        def candidate_to_BLOCK(candidate : list):
            return [BLOCK(self.player,fence[0],fence[1]) for fence in candidate]

        def generate_neighbor(candidate,applicable_fence,FC,FH,FV):
            copied_candidate = deepcopy(candidate)
            copied_FC = deepcopy(FC)
            copied_FH = deepcopy(FH)
            copied_FV = deepcopy(FV)
            applicable_fence = [fence for fence in applicable_fence if (fence not in copied_candidate) and (check_valid(fence[0],fence[1],FC,FH,FV))]
            n = randint(1,min(5,len(applicable_fence)))
            self._logger.debug(f"n : {n}")
            for _ in range(n):
                self._logger.debug(f"len copied candidate : {len(copied_candidate)}")
                pop = copied_candidate.pop()
                copied_FC.remove(list(pop[0]))
                if pop[1]=='horizontal':
                    copied_FH.remove(list(pop[0]))
                if pop[1]=='vertical':
                    copied_FV.remove(list(pop[0]))

            count = 0
            result = []
            while count < n:
                self._logger.debug(f"len applicable_fence : {len(applicable_fence)}")
                new_cand = choice(applicable_fence)
                if check_valid(new_cand[0],new_cand[1],copied_FC,copied_FH,copied_FV):
                    result.append(new_cand)
                    self._logger.debug(f"new_cand[0] : {new_cand[0]}")
                    copied_FC.append(list(new_cand[0]))
                    if new_cand[1]=='horizontal':
                        copied_FH.append(list(new_cand[0]))
                    if new_cand[1]=='vertical':
                        copied_FV.append(list(new_cand[0]))
                    count += 1
                if len(applicable_fence)==1:
                    break
                else:
                    applicable_fence.remove(new_cand)
            return copied_candidate + result

        def obj(fences, opponent, opponent_pos):
            # self._logger.debug("compute obj..")
            # state = board.simulate_action(None,*candidate_to_BLOCK(fence))
            # self._logger.debug("simulate done")
            # return board.distance_to_goal(opponent,state)
            obj_val = 0
            for fence in fences:
                if fence[1]=='horizontal':
                    cur_pos = fence[0]
                    nxt_pos = (cur_pos[0]+1,cur_pos[1])
                    obj_val += 2*board.get_move_turns(cur_pos,nxt_pos)
                if fence[1]=='vertical':
                    cur_pos = fence[0]
                    nxt_pos = (cur_pos[0],cur_pos[1]+1)
                    obj_val += board.get_move_turns(cur_pos,nxt_pos)
                distance_to_opponent = -int(math.sqrt((opponent_pos[1]-cur_pos[1])**2)) # <- only column distance / both -> int(math.sqrt((opponent_pos[0]-cur_pos[0])**2 + (opponent_pos[1]-cur_pos[1])**2))
                obj_val += 6*distance_to_opponent
            return obj_val 
        
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
        
        total_count = 0
        count = 0
        reached = [candidate]
        while True:
            if count == 10 or total_count == 400:
                break
            shuffle(candidate)
            neighbor = generate_neighbor(candidate,board.get_applicable_fences(),fenceCenter,fenceHorizontal,fenceVertical) # FC,FH,FVÃªÂ°Â updateÃ«ÂÂÃ¬ÂÂ´Ã¬ÂÂ¼ Ã­ÂÂ¨. Ã¬ÂÂ¬ÃªÂ¸Â° Ã¬ÂÂÃ¬Â Â.
            self._logger.debug(f"neighbor is {neighbor}")
            self._logger.debug(f"current is {candidate}")

            setA = {fence for fence in candidate}
            setB = {fence for fence in neighbor}
            self._logger.debug(f"difference is {setA.difference(setB)}, len = {len(setA.difference(setB))}")
            #neighbor = [((4,3),'horizontal'),((6,7),'horizontal'),((2,7),'horizontal'),((3,7),'horizontal'),((6,4),'horizontal'),((5,1),'horizontal')]
            
            current_obj = obj(candidate, opponent, opponent_position)
            self._logger.debug(f"cur obj : {current_obj}")
            neighbor_obj = obj(neighbor, opponent, opponent_position)
            self._logger.debug(f"cur obj : {current_obj}, neighbor obj : {neighbor_obj}")
            if neighbor_obj >= current_obj: # Ã¬ÂÂ¼Ã«ÂÂ¨Ã¬ÂÂ side walk Ã«Â¶ÂÃªÂ°ÂÃ«ÂÂ¥ Ã¬Â¡Â°ÃªÂ±Â´Ã¬ÂÂ¼Ã«Â¡Â Ã«ÂÂ . Ã¬ÂÂ´Ã­ÂÂ reached Ã«Â§ÂÃ«ÂÂ¤Ã¬ÂÂ´Ã¬ÂÂ side walk Ã«Â§ÂÃ«ÂÂ¤Ã¬ÂÂ´Ã«ÂÂ Ã«ÂÂ  Ã«ÂÂ¯
                for prev in reached:
                    if set(neighbor)==set(prev):
                        count += 1
                        total_count += 1
                        continue
                self._logger.debug(f"change to neighbor : {candidate} -> {neighbor}")
                candidate, fenceCenter, fenceHorizontal, fenceVertical = change_to_neighbor(candidate,neighbor,fenceCenter, fenceHorizontal, fenceVertical)
                count = 0
                reached.append(neighbor)
            total_count += 1
        self._logger.debug(f"board.distance_to_goal(opponent) : {board.distance_to_goal(opponent)}")
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
            
        #     # Ã­ÂÂÃ«Â²Â Move Ã«Â¥Â¼ Ã«Â°ÂÃ­ÂÂÃ­ÂÂÃªÂ³Â  Ã«ÂÂÃ«Â©Â´ reachedÃ«ÂÂ Ã¬Â´ÂÃªÂ¸Â°Ã­ÂÂ Ã«ÂÂÃ«Â¯ÂÃ«Â¡Â Ã«ÂÂÃ¬ÂÂ¼Ã­ÂÂ obj ÃªÂ°ÂÃ¬ÂÂ ÃªÂ°ÂÃ¬Â§ÂÃ«ÂÂ Ã¬ÂÂÃ¬Â¹ÂÃ«Â¥Â¼ Ã¬ÂÂÃ«ÂÂ¤ ÃªÂ°ÂÃ«ÂÂ¤ Ã«Â°ÂÃ«Â³ÂµÃ­ÂÂ¨.
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
        from collections import deque
        from heapq import heappop, heappush
        import time

        start_time = time.time()
        TIME_BUFFER = 1.0  # 1초 남기고 종료

        player = board.current_player()
        opponent = 'white' if player == 'black' else 'black'
        max_depth = 4
        
        def get_all_actions(board,state, player):
            board.set_to_state(state)
            move_positions = board.get_applicable_moves(player)
            move_actions = [MOVE(player=player, position=pos) for pos in move_positions]
            return move_actions

        def shortest_route_bfs(state, player):
            start = state['player'][player]['pawn']
            goal_row = 0 if player == 'black' else 8

            fence = state['board']['fence_center']
            fh = set()
            fv = set()
            for f in fence:
                if f[1] == 'h':
                    fh.add(f[0]) 
                else:
                    fv.add(f[0])

            visited = set()
            queue = deque([(start, 0)])  # (position, turn count)

            while queue:
                (r, c), turn = queue.popleft()

                if (r, c) in visited:
                    continue
                visited.add((r, c))

                if r == goal_row:
                    return turn

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr <= 8 and 0 <= nc <= 8):
                        continue

                    blocked = False
                    if dr == 1 and (((r, c) in fh) or ((r + 1, c-1) in fh)):
                        blocked = True
                    elif dr == -1 and (((r-1, c) in fh) or ((r-1, c-1) in fh)):
                        blocked = True
                    elif dc == 1 and (((r, c) in fv) or ((r-1, c) in fv)):
                        blocked = True
                    elif dc == -1 and (((r, c-1) in fv) or ((r-1, c-1) in fv)):
                        blocked = True

                    if not blocked and (nr, nc) not in visited:
                        queue.append(((nr, nc), turn + 1))
            return float('inf')
        
        def evaluate(state):
            opponent = 'white' if self.player == 'black' else 'black'
            my_turn = shortest_route_bfs(state,self.player)
            opp_turn = shortest_route_bfs(state, opponent)
            if self.player == 'white':
                my_turn += 1
            return opp_turn - 2*my_turn

        def fence_expanshion_operator(state):
            phase = 10 - state['player'][self.player]['fences_left']
            opponent = 'white' if self.player == 'black' else 'black'
            opp_pos = state['player'][opponent]['pawn']
            my_pos = state['player'][self.player]['pawn']
            FC = state['board']['fence_center']

            if self.player == 'black':
                if phase == 0:
                    return ((7,3),'horizontal')
                if phase == 1:
                    if opp_pos[1] >= 4:
                        if ((7,5), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((7,5), 'horizontal')
                        if ((4,6), 'vertical') in board.get_applicable_fences(self.player):
                            return ((4,6), 'vertical')
                        return ((6,2),'vertical')
                    else :
                        return ((6,2),'vertical')
                if phase == 2:
                    if opp_pos[1] >= 4:
                        if ((7,5), 'h') in FC:
                            return ((6,2),'vertical')
                        if ((5,5), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((5,5), 'horizontal')
                        if ((7,7), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((7,7), 'horizontal')
                        return (my_pos,'horizontal')
                    else:
                        if ((5,2), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((5,2), 'horizontal')
                        else :
                            return ((7,1), 'horizontal')
                if phase == 3:
                    if opp_pos[1] >= 4:
                        if ((7,7), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((7,7), 'horizontal')
                        if ((6,2), 'vertical') in board.get_applicable_fences(self.player):
                            return ((6,2), 'vertical')
                        if ((4,2), 'vertical') in board.get_applicable_fences(self.player):
                            return ((4,2),'vertical')
                        if ((2,2), 'vertical') in board.get_applicable_fences(self.player):
                            return ((2,2),'vertical')
                        return (my_pos,'horizontal')
                    else :
                        if ((5,2), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((5,2), 'horizontal')
                        if ((5,0), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((5,0), 'horizontal')
                        if ((6,1), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((6,1), 'horizontal')
                        return ((7,1), 'horizontal')
                return ((7,5), 'horizontal')
            else :
                if phase == 0:
                    return ((0, 4), 'horizontal')

                if phase == 1:
                    if opp_pos[1] <= 4:
                        if ((0, 2), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((0, 2), 'horizontal')
                        elif ((3, 1), 'vertical') in board.get_applicable_fences(self.player):
                            return ((3, 1), 'vertical')
                        else:
                            return ((1, 5), 'vertical')
                    else:
                        return ((1, 5), 'vertical')

                if phase == 2:
                    if opp_pos[1] <= 4:
                        if ((0, 2), 'h') in FC:
                            return ((1, 5), 'vertical')
                        if ((2, 2), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((2, 2), 'horizontal')
                        if ((0, 0), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((0, 0), 'horizontal')
                        return ((7 - my_pos[0], 7 - my_pos[1]), 'horizontal')
                    else:
                        if ((2, 5), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((2, 5), 'horizontal')
                        else:
                            return ((0, 6), 'horizontal')

                if phase == 3: 
                    if opp_pos[1] <= 4:
                        if ((3, 5), 'vertical') in board.get_applicable_fences(self.player):
                            return ((3, 5), 'vertical')
                        if ((0, 0), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((0, 0), 'horizontal')
                        if ((1, 5), 'vertical') in board.get_applicable_fences(self.player):
                            return ((1, 5), 'vertical')
                        return ((7 - my_pos[0], 7 - my_pos[1]), 'horizontal')
                    else:
                        if ((3, 5), 'vertical') in board.get_applicable_fences(self.player):
                            return ((3, 5), 'vertical')
                        if ((2, 5), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((2, 5), 'horizontal')
                        elif ((1, 6), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((1, 6), 'horizontal')
                        return ((0, 6), 'horizontal')
                if phase == 4:
                    if opp_pos[1] <= 4:
                        if ((0, 0), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((0, 0), 'horizontal')
                    else:
                        if opp_pos[1] == 5 and ((5, 5), 'vertical') in board.get_applicable_fences(self.player):
                            return ((5, 5), 'vertical')
                return ((0, 0), 'horizontal')
            
             # 순서 바꿔야 함

        def alpha_beta_search(state, depth, alpha, beta, maximizing):
            if time.time() > time_limit - TIME_BUFFER:
                raise TimeoutError()

            current = player if maximizing else opponent
            actions = [MOVE(current, m) for m in board.get_applicable_moves(current)]

            if depth == 0 or board.is_game_end():
                return evaluate(state), None

            best_action = None
            if maximizing:
                max_eval = float('-inf')
                for act in actions:
                    try:
                        new_state = board.simulate_action(state, act)
                        score, _ = alpha_beta_search(new_state, depth - 1, alpha, beta, False)
                        #self._logger.debug(f"max - act : {act} / eval : {score} / max_eval : {max_eval}")
                        if score > max_eval:
                            max_eval = score
                            best_action = act
                        alpha = max(alpha, score)
                        # if beta <= alpha:
                        #     break
                    except Exception as e:
                        continue
                return max_eval, best_action
            else:
                min_eval = float('inf')
                for act in actions:
                    try:
                        new_state = board.simulate_action(state, act)
                        score, _ = alpha_beta_search(new_state, depth - 1, alpha, beta, True)
                        #self._logger.debug(f"min - act : {act} / eval : {score} / min_eval : {min_eval}")
                        if score < min_eval:
                            min_eval = score
                            best_action = act
                        beta = min(beta, score)
                        # if beta <= alpha:
                        #     break
                    except Exception as e:
                        continue
                return min_eval, best_action
        
        def valid_move(pos, fence):
            if self.player == 'black':
                if (((pos[0]-1,pos[1]-1),'h') in fence) or (((pos[0]-1,pos[1]),'h') in fence):
                    return False
                return True
            else:
                if ((pos,'h') in fence) or (((pos[0],pos[1]-1),'h') in fence):
                    return False
                return True
        
        current_state = board.get_state()
        value, move = alpha_beta_search(board.get_state(), max_depth, float('-inf'), float('inf'), True)
        cur_pos = current_state['player'][self.player]['pawn']
        opponent = 'white' if self.player =='black' else 'black'
        goal = 8 if self.player == 'white' else 0
    
        block = None
        if shortest_route_bfs(current_state, opponent) <= 6:
            block = fence_expanshion_operator(current_state)
        #self._logger.debug(board.get_applicable_fences(self.player))
        current_fence = board.get_state()['board']['fence_center']
        valid_fence = board.get_applicable_fences(self.player)
        if block not in valid_fence:
            if self.player == 'black' and cur_pos[0]-1 == goal and valid_move(cur_pos,current_fence):
                return MOVE(self.player,(cur_pos[0]-1, cur_pos[1]))
            if self.player == 'white' and cur_pos[0]+1 == goal and valid_move(cur_pos,current_fence):
                return MOVE(self.player,(cur_pos[0]+1, cur_pos[1]))
            return move
        else :
            return BLOCK(self.player, block[0], block[1])
    

