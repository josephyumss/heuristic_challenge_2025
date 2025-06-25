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
        def check_valid(pos,ori,FC,FH,FV):
            if list(pos) in FC: 
                return False
                        
            if ori == "h":
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
            if ori == 'v':
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
        
        def get_all_actions(board, player):
            # pawn moves
            self._logger.debug(f"\nturn : {board.current_player()}")
            self._logger.debug(f"position : {board.get_position(player)}")
            self._logger.debug(f"current_state : {board.get_state()}")
            move_positions = board.get_applicable_moves(player)
            move_actions = [MOVE(player=player, position=pos) for pos in move_positions]
            self._logger.debug(move_positions)
            self._logger.debug(move_actions)
            current_state=board.get_state()

            fc = current_state['board']['fence_center']
            fh = []
            fv = []
            for fence_center in fc:
                if fence_center[1]=='h':
                    fh.append(fence_center)

                if fence_center[1]=='v':
                    fv.append(fence_center)
            # fence placements
            fence_actions = []
            if board.number_of_fences_left(player) > 0:
                fence_positions = board.get_applicable_fences(player)
                for (r, c), o in fence_positions:
                    if check_valid((r,c),o,fc,fh,fv):
                        fence_actions.append(BLOCK(player=player, orientation=o, edge=(r, c)))
            return move_actions + fence_actions

        def eval(board):
            opponent = 'white' if self.player == 'black' else 'black'

            my_pos = board.get_position(self.player)
            opp_pos = board.get_position(opponent)

            my_goal_row = 8 if self.player == 'white' else 0
            opp_goal_row = 0 if self.player == 'white' else 8

            if board.is_game_end():
                if my_goal_row == my_pos[0]:
                    return 999
                else :
                    return -999

            my_dist = abs(my_goal_row - my_pos[0])
            opp_dist = abs(opp_goal_row - opp_pos[0])

            fences = current_state['board']['fence_center']
            fence_behind_me_count = 0
            for fence in fences:
                if self.player=='black':
                    if fence[0]>my_pos[0]:
                        fence_behind_me_count += 1
                else :
                    if fence[0]<my_pos[0]:
                        fence_behind_me_count += 1

            # my_fences = board.number_of_fences_left(player)
            # opp_fences = board.number_of_fences_left(opponent)
            #self._logger.debug(f"fence_behind me : {fence_behind_me_count}")
            return fence_behind_me_count - my_dist #+ 0.1 * (my_fences - opp_fences)

        def maxValue(board, state, player, alpha, beta, depth):
            DEPTH_LIMIT = 2
            #self._logger.debug("working...")
            if player == 'black' : opponent = 'white'
            else : opponent = 'black'

            try:
                board.set_to_state(state)
            except:
                return eval(board), None

            if board.is_game_end() or depth == DEPTH_LIMIT:
                return eval(board), None
            
            v, move = -float('inf'), None
            #self._logger.debug(get_all_actions(board, player))
            for act in get_all_actions(board, player):           
                try:
                    board.set_to_state(state)
                    cur_state = board.get_state()
                    simulate_state = board.simulate_action(cur_state, act)
                except :
                    continue

                v2, a2 = minValue(board, simulate_state, opponent, alpha, beta, depth+1)
                if v2 > v :
                    self._logger.debug(f"v : {v}, v2 : {v2}")
                    v, move = v2, act
                    alpha = max(alpha, v)
                if v > beta :
                    return v, move

            return v, move

        def minValue(board, state, player, alpha, beta, depth):
            DEPTH_LIMIT = 4
            if player == 'black' : opponent = 'white'
            else : opponent = 'black'

            try:
                board.set_to_state(state)
            except :
                return eval(board), None
            
            if board.is_game_end() or depth == DEPTH_LIMIT:
                return eval(board), None
            
            v, move = float('inf'), None

            for act in get_all_actions(board, player):
                try:
                    board.set_to_state(state)
                    cur_state = board.get_state()
                    simulate_state = board.simulate_action(cur_state, act)
                except :
                    continue

                v2, a2 = maxValue(board, simulate_state, opponent, alpha, beta, depth+1)
                
                if v2 < v :
                    v, move = v2, a2
                    beta = min(beta, v)

                if v < alpha :
                    return v, move
            return v, move

        current_state = board.get_state()
        self._logger.debug(f"from main , applicable moves : {board.get_applicable_moves(self.player)}")
        value, move = maxValue(board, current_state, self.player, -float('inf'), float('inf'), 0)
        # if move == None :
        #     move = choice(board.get_applicable_moves(self.player))
        #     move = MOVE(self.player,move)
        return move
    
