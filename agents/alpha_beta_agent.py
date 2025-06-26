from pathlib import Path
from random import choice
from time import time
from traceback import print_exc, format_exc
from typing import Tuple, List, Callable, Dict, Generator, Optional, Literal, Union
from action import *
from board import GameBoard
from datetime import datetime
import random
import logging

class Agent:  # Do not change the name of this class!
    """
    An agent class
    """

    # Do not modify this.
    name = Path(__file__).stem
    # _logger = logging.getLogger('Alpha-Beta Agent')
    # _logger.setLevel(logging.DEBUG) 

    # Do not change the constructor argument!
    def __init__(self, player: Literal['white', 'black']):
        """
        Initialize the agent

        :param player: Player label for this agent. White or Black
        """
        self.player = player
        self._logger = logging.getLogger('MyAgent')
        self._logger.setLevel(logging.DEBUG)  

        if not self._logger.hasHandlers():  
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._logger.debug("Agent logger initialized")

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
        raise NotImplementedError()

    def local_search(self, board: GameBoard, time_limit: float) -> List[BLOCK]:
        """
        * Complete this function to answer the challenge PART II.

        This function uses local search for finding the three best place of the fence.
        The system calls your algorithm multiple times, repeatedly.
        Each time, it provides new position of your pawn and asks your next decision
         until time limit is reached, or until you return a BLOCK action.

        Each time you have to decide one of the action.
        - If you want to look around neighborhood places, return MOVE action
        - If you decide to answer the best place, return BLOCK action
        * Note: you cannot move to the position that your opponent already occupied.

        You can use your heuristic search function, which is previously implemented, to compute the fitness score of each place.
        * Note that we will not provide any official fitness function here. The quality of your answer depends on the first part.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - Hill-climbing search and its variants
        - Simulated annealing and its variants
        - Tabu search and its variants
        - Greedy Best-first search
        - Local/stochastic beam search (note: parallel execution should be called as sequentially)
        - Evolutionary algorithms
        - Empirical/Stochastic gradient methods
        - Newton-Raphson method

        :param board: The game board with current state.
        :param time_limit: The time limit for the search. Datetime.now() should have lower timestamp value than this.
        :return: The list of three BLOCKs. That is, you should return [BLOCK(), BLOCK(), BLOCK()].
        """
        raise NotImplementedError()

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
        from collections import deque
        from heapq import heappop, heappush
        import time

        start_time = time.time()
        TIME_BUFFER = 1.0

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
            queue = deque([(start, 0)])

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
            my_turn = shortest_route_bfs(state,self.player)
            if self.player == 'black':
                my_turn += 1
            
            #self._logger.debug(f"turn : {my_turn}, pos : {state['player'][self.player]['pawn']}")
            return -my_turn

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
                if phase >= 4 :
                    if (opp_pos[1] >= 3) and (opp_pos[0] >= 4) and ((4,2), 'vertical') in board.get_applicable_fences(self.player):
                        return ((4,2), 'vertical')
                    if (opp_pos[1] >= 3) and (opp_pos[0] >= 2) and ((2,2), 'vertical') in board.get_applicable_fences(self.player):
                        return ((2,2), 'vertical')
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
                if phase >= 4:
                    if (opp_pos[0]==1) and (opp_pos[1] <= 4) and (((2,7),'h') not in FC):
                        if ((0, 0), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((0, 0), 'horizontal')
                    else:
                        self._logger.debug((((5, 5), 'vertical') in board.get_applicable_fences(self.player)))
                        self._logger.debug(((opp_pos[1] == 6) or (opp_pos[1] ==5)))
                                           
                        if ((opp_pos[1] == 6) or (opp_pos[1] ==5)) and (((5, 5), 'vertical') in board.get_applicable_fences(self.player)):
                            return ((5, 5), 'vertical')
                        if ((2, 5), 'horizontal') in board.get_applicable_fences(self.player):
                            return ((2, 5), 'horizontal')
                        if (((2, 7), 'horizontal') in board.get_applicable_fences(self.player)) and (((0,0),'h') not in FC) and (opp_pos[1] > 5):
                            return ((2, 7), 'horizontal')                       
                return ((0, 0), 'horizontal')

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
                        if score > max_eval:
                            max_eval = score
                            best_action = act
                        alpha = max(alpha, score)

                    except Exception as e:
                        continue
                return max_eval, best_action
            else:
                min_eval = float('inf')
                for act in actions:
                    try:
                        new_state = board.simulate_action(state, act)
                        score, _ = alpha_beta_search(new_state, depth - 1, alpha, beta, True)
                        if score < min_eval:
                            min_eval = score
                            best_action = act
                        beta = min(beta, score)
                    except Exception as e:
                        continue
                return min_eval, best_action
        
        def valid_move(pos, opp_pos, fence):
            if self.player == 'black':
                
                if (((pos[0]-1,pos[1]-1),'h') in fence) or (((pos[0]-1,pos[1]),'h') in fence) or (pos[0]-1, pos[1])==opp_pos:
                    return False
                return True
            else:
                # self._logger.debug(f"(pos[0]-1, pos[1]) : {(pos[0]+1, pos[1])}")
                # self._logger.debug(f"opp_pos : {opp_pos}")
                if ((pos,'h') in fence) or (((pos[0],pos[1]-1),'h') in fence) or (pos[0]+1, pos[1])==opp_pos:
                    return False
                return True
        
        current_state = board.get_state()
        value, move = alpha_beta_search(board.get_state(), max_depth, float('-inf'), float('inf'), True)
        cur_pos = current_state['player'][self.player]['pawn']
        opponent = 'white' if self.player =='black' else 'black'
        goal = 8 if self.player == 'white' else 0
    
        block = None
        my_turn = shortest_route_bfs(current_state, self.player)
        opp_turn = shortest_route_bfs(current_state, opponent)
        if (my_turn >= opp_turn-2) and abs(goal-cur_pos[0])<=7:
            block = fence_expanshion_operator(current_state)
        current_fence = board.get_state()['board']['fence_center']
        valid_fence = board.get_applicable_fences(self.player)
        opp_pos = current_state['player'][opponent]['pawn']
        if block not in valid_fence:
            if self.player == 'black' and cur_pos[0]-1 == goal and valid_move(cur_pos,opp_pos,current_fence):
                return MOVE(self.player,(cur_pos[0]-1, cur_pos[1]))
            if self.player == 'white' and cur_pos[0]+1 == goal and valid_move(cur_pos,opp_pos,current_fence):
                return MOVE(self.player,(cur_pos[0]+1, cur_pos[1]))
            return move
        else :
            return BLOCK(self.player, block[0], block[1])
        
        
    
