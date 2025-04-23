# Logging method for board execution
import heapq
import logging
# Library for OS environment
import os
import random
import sys
# Object-level deep copy method
from copy import deepcopy
# Random number generators
from random import randint as random_integer
# Type specification for Python code
from typing import Tuple, List, Literal

# Process information class: for memory usage tracking
from psutil import Process as PUInfo, NoSuchProcess
# Import some class definitions that implements the Settlers of Catan game.
from pyquoridor.board import Board
from pyquoridor.exceptions import GameOver, InvalidFence, InvalidMove  # InvalidFence 추가
from pyquoridor.square import MAX_COL, MAX_ROW

# Import action specifications
from action import Action, BLOCK, MOVE, FENCES_MAX
# Import some utilities
from util import print_board

#: True if the program run with 'DEBUG' environment variable.
IS_DEBUG = '--debug' in sys.argv
IS_RUN = 'fixed_evaluation' in sys.argv[0]

class HeuristicAgent:
    def heuristic(self, current_row: int, target_row: int, current_col: int, board_width=5) -> float:
        rowDist = abs(current_row - target_row)

        # How far from the center column
        colCenter = board_width // 2
        colDist = abs(current_col - colCenter)

        # Mainly considering row distance, including the column distance slightly.
        return rowDist + 0.1 * colDist

    def heuristic_search(self, board: "GameBoard", player: str) -> int:
        # Initialize
        initial_state = board.get_state()
        target_row = 8 if player else 0

        initial_pos = tuple(initial_state['player'][player]['pawn'])
        initial_id = initial_state['state_id']

        came_from = {}
        g_score = {initial_id: 0}
        states = {initial_id: initial_state}

        h_init = self.heuristic(initial_pos[0], target_row, initial_pos[1])
        open_set = [(h_init, initial_id)]

        visited_positions = {initial_pos: 0}

        board.set_to_state(initial_state)

        while open_set:
            _, current_id = heapq.heappop(open_set)

            current_state = states[current_id]
            board.set_to_state(current_state)

            current_pos = tuple(current_state['player'][player]['pawn'])
            current_turns = g_score[current_id]
            current_row, current_col = current_pos

            if current_row == target_row:
                return current_turns

            for next_pos in board.get_applicable_moves(player):
                move_cost = board.get_move_turns(current_pos, next_pos)
                new_turns = current_turns + move_cost

                if (next_pos in visited_positions
                        and visited_positions[next_pos] <= new_turns):
                    continue

                move = MOVE(player, next_pos)
                next_state = board.simulate_action(current_state, move, problem_type=1)
                next_id = next_state['state_id']

                came_from[next_id] = (current_id, move)
                g_score[next_id] = new_turns
                visited_positions[next_pos] = new_turns
                states[next_id] = next_state

                # New heuristic
                h_val = self.heuristic(next_pos[0], target_row, next_pos[1])
                f_val = new_turns + h_val
                heapq.heappush(open_set, (f_val, next_id))

        return 0

class GameBoard:
    """
    The game board object.
    By interacting with Board, you can expect what will happen afterward.
    """
    #: [PRIVATE] The game instance running currently. Don't access this directly in your agent code!
    _board: Board = None
    #: [PRIVATE] Your side (black/white). Don't access this directly in your agent code!
    _player_side = 'black'
    #: [PRIVATE] The current player's index.
    _current_player = 'white'
    #: [PRIVATE] The initial state of the board. Don't access this directly in your agent code!
    _initial = None
    #: [PRIVATE] The current state of the board. Don't access this directly in your agent code!
    _current = None
    #: [PRIVATE] Logger instance for Board's function calls
    _logger = logging.getLogger('GameBoard')
    #: [PRIVATE] Memory usage tracker
    _process_info = None
    #: [PRIVATE] Fields for computing maximum memory usage. Don't access this directly in your agent code!
    _init_memory = 0
    _max_memory = 0
    _heuristic_calls = 0
    _initial_fences_dict = {}
    #: [PRIVATE] Random seed generator
    _rng = random.Random(2938)

    def _initialize(self, start_with_random_fence: int = 0):
        """
        Initialize the board for evaluation. ONLY for evaluation purposes.
        [WARN] Don't access this method in your agent code.
        """
        # Initialize process tracker
        self._process_info = PUInfo(os.getpid())

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Initializing a new game board...')
        
        # Initialize a new game board
        self._board = Board()

        self._vertical_turns = [[self._rng.randint(1, 5) for _ in range(9)] for _ in range(8)]
        self._horizontal_turns = [[self._rng.randint(1, 5) for _ in range(8)] for _ in range(9)]

        # Initialize board renderer for debugging purposes
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Rendered board: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))

        # Pick a starting point randomly.
        self._player_side = random.choice(['black', 'white'])
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'You\'re player {self._player_side}')

        for p in ['black', 'white']:
            row = self._board.pawns[p].square.row
            col = random_integer(0, MAX_COL - 1)
            self._board.pawns[p].move(self._board.get_square_or_none(row, col))

        self._board.turn = 0 if self._player_side == 'white' else 1
        # Update position information with a new starting point
        for pawn in self._board.pawns.values():
            pawn.square.reset_neighbours()

        for pawn in self._board.pawns.values():
            self._board.update_neighbours(pawn.square)

        # Set random fences
        assert start_with_random_fence < 5, 'Do not use start_with_random_fence >= 5'
        for _ in range(start_with_random_fence):
            for p in self._board.pawns.keys():
                fences = self.get_applicable_fences(p)
                while fences:
                    fence, orientation = self._rng.choice(fences)
                    try:
                        BLOCK(p, fence, orientation)(self)
                        break
                    except InvalidFence:
                        fences.remove((fence, orientation))
                        continue
                    except:
                        raise

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('After moving initial position: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))

        # Store initial state representation
        self._initial = self._save_state()
        self._current = deepcopy(self._initial)

        # Update memory usage
        self._update_memory_usage()

    def get_move_turns(self, current_pos: tuple, next_pos: tuple) -> int:
        """Return the number of turns required to move between adjacent positions"""
        row1, col1 = current_pos
        row2, col2 = next_pos
        min_row = min(row1, row2)
        min_col = min(col1, col2)
        if col1 == col2:
            return self._vertical_turns[min_row][col1]
        elif row1 == row2:
            return self._horizontal_turns[row1][min_col]
        else:
            return max(self._horizontal_turns[min_row][min_col], self._vertical_turns[min_row][min_col])
    
    def print_turns(self):
        """Print the required turns for each edge in a visual format"""
        for i in range(9):
            row = ""
            for j in range(8):
                row += f"O-{self._horizontal_turns[i][j]}-"
            row += "O"
            print(row)
            
            if i < 8:
                row = ""
                for j in range(9):
                    row += f"{self._vertical_turns[i][j]}    "
                print(row)
        
    def reset_memory_usage(self):
        """
        Reset memory usage
        """
        self._init_memory = 0
        self._max_memory = 0
        self._heuristic_calls = 0
        self._update_memory_usage()

    def set_to_state(self, specific_state=None, is_initial: bool = False):
        """
        Restore the board to the initial state for repeated evaluation.
        :param specific_state: A state representation which the board reset to
        :param is_initial: True if this is an initial state to begin evaluation
        """
        assert specific_state is not None or not is_initial
        if specific_state is None:
            specific_state = self._initial
        if is_initial:
            self._initial = specific_state
            self._current = deepcopy(self._initial)
            self._rng.seed(hash(self._initial['state_id']))  # Use state_id as hash seed.
        else:
            self.check_state_difference(self._current, specific_state)

        # Restore the board to the given state.
        self._restore_state(specific_state)

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('State has been set as follows: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))

    def _set_initial_fences(self, fences):
        """
        Store the initial state of the fences
        """
        self._initial_fences_dict = fences

    def is_game_end(self):
        """
        Check whether the given state indicate the end of the game

        :return: True if the game ends at the given state
        """
        is_game_end = self._board.game_finished()
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'Querying whether the game ends in this state... Answer = {is_game_end}')
        return is_game_end

    def get_state(self) -> dict:
        """
        Get the current board state
        현재 state 반환

        :return: A copy of the current board state dictionary
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying current state...')

        # Check whether the game has been initialized or not.
        assert self._current is not None, 'The board should be initialized. Did you run the evaluation code properly?'
        # Return the initial state representation as a copy.
        return deepcopy(self._current)

    def get_initial_state(self) -> dict:
        """
        Get the initial board state
        :return: A copy of the initial board state dictionary
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying initial state...')

        # Check whether the game has been initialized or not.
        assert self._initial is not None, 'The board should be initialized. Did you run the evaluation code properly?'
        # Return the initial state representation as a copy.
        return deepcopy(self._initial)

    def get_player_id(self) -> Literal['black', 'white']:
        """
        Return the player name.
        """
        return self._player_side

    def get_opponent_id(self) -> Literal['black', 'white']:
        """
        Return the opponent name.
        """
        return 'black' if self._player_side == 'white' else 'white'

    def get_applicable_moves(self, player: Literal['black', 'white'] = None) -> List[Tuple[int, int]]:
        """
        Get the list of applicable roads
        :param player: Player name. black or white. (You can ask your player ID by calling get_player_index())
        :return: A copy of the list of applicable move coordinates.
            (List of Tuple[int, int].)
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying applicable move directions...')

        # Read all applicable positions
        player = self._board.current_player() if player is None else player
        applicable_positions = sorted([
            (square.row, square.col)
            for square in self._board.valid_pawn_moves(player, check_winner=False)
        ])

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'List of applicable move positions: {applicable_positions}')

        # Return applicable positions as list of tuples.
        return applicable_positions

    def get_applicable_fences(self, player: Literal['black', 'white'] = None)\
            -> List[Tuple[Tuple[int, int], Literal['horizontal', 'vertical']]]:
        """
        Get the list of applicable fences

        :param player: Player name. black or white. (You can ask your player ID by calling get_player_index())
        :return: A copy of the list of applicable fence coordinates with its orientation (horizontal or vertical).
            (List of Tuple[Tuple[int, int], str].)
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying applicable move directions...')

        # Read all applicable positions
        player = self._board.current_player() if player is None else player
        if self._board.fences_left[player] == 0:
            if IS_DEBUG:  # Logging for debug
                self._logger.debug(f'{player} used all fences.')
            return []

        applicable_fences = []
        for r in range(MAX_ROW - 1):
            for c in range(MAX_COL - 1):
                # Pass positions whose center of a grid.
                if self._board.fence_center_grid[r, c]:
                    continue

                if not self._board.fence_exists(r, c, 'h'):
                    applicable_fences.append(((r, c), 'horizontal'))
                if not self._board.fence_exists(r, c, 'v'):
                    applicable_fences.append(((r, c), 'vertical'))

        applicable_fences = sorted(applicable_fences)

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'List of applicable fence positions: {applicable_fences}')

        # Return applicable positions as list of tuples.
        return applicable_fences

    def get_current_memory_usage(self):
        """
        :return: Current memory usage for the process having this board
        """
        try:
            usage = self._process_info.memory_full_info().uss
            if self._init_memory == 0:
                self._init_memory = usage
            return usage
        except NoSuchProcess:
            if self._max_memory >= 0:
                self._logger.warning('As tracking the process has been failed, '
                                     'I turned off memory usage tracking ability.')
                self._max_memory = -1
            return -1

    def get_max_memory_usage(self):
        """
        :return: Maximum memory usage for the process having this board
        """
        return max(0, self._max_memory - self._init_memory)

    def _update_memory_usage(self):
        """
        [PRIVATE] updating maximum memory usage
        """
        if self._max_memory >= 0:
            self._max_memory = max(self._max_memory, self.get_current_memory_usage())

    def number_of_fences_left(self, player: Literal['black', 'white']):
        """
        Number of fences remained as uninstalled.
        :param player: Player name to compute the number of fences unused
        :return: (int) The number of unused fences
        """
        return self._board.fences_left[player]

    def check_state_difference(self, state_from, state_to):
        """
        Check whether the state transition is allowed within the current problem setting (Challenge II)
        If not, the function will raise an Exception.
        :param state_from: The previous state moving from
        :param state_to: The next state moving to
        """

        prev_fences = {tuple(r) for r in state_from['board']['fence_center']}
        next_fences = {tuple(r) for r in state_to['board']['fence_center']}
        fences_removed = len(prev_fences.difference(next_fences))

        assert fences_removed <= 3, f'More than 3 fences were removed: {fences_removed} fences'

    def distance_to_goal(self, player: Literal['black', 'white']):
        """
        Compute distance toward the goal line.
        :param player: Player name to compute the distance toward the goal line.
        :return: (int) Total distance (counting turns of movement)
        """
        self._heuristic_calls += 1
        agent = HeuristicAgent()
        return agent.heuristic_search(self, player)
    
    def simulate_action(self, state: dict = None, *actions: Action, problem_type: int = 4) -> dict:
        """
        Simulate given actions.

        Usage:
            - `simulate_action(state, action1)` will execute a single action, `action1`
            - `simulate_action(state, action1, action2)` will execute two consecutive actions, `action1` and `action2`
            - ...
            - `simulate_action(state, *action_list)` will execute actions in the order specified in the `action_list`

        :param state: State where the simulation starts from. If None, the simulation starts from the initial state.
        :param actions: Actions to simulate or execute.
        :param problem_type: Problem Type Index (Challenge problem #)
        :return: The last state after simulating all actions
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'------- SIMULATION START: {actions} -------')

        # Restore to the given state
        self.set_to_state(state)

        for act in actions:  # For each actions in the variable arguments,
            # Run actions through calling each action object. If error occurs, raise as it is (except for GameOver)
            try:
                # For challenge I and II, force the current player to agents
                if problem_type == 1:
                    self._board.turn = 0 if act.player == 'white' else 1
                if problem_type == 2:
                    self._board.turn = 0 if self._player_side == 'white' else 1

                act(self, avoid_check=problem_type == 2)
            except GameOver:
                break
            except InvalidMove as e:
                if problem_type != 2:
                    raise e

            # Break the loop if the game ends within executing actions.
            if self.is_game_end():
                break

        # Copy the current state to return
        if problem_type < 3:
            self._board.turn = 0 if self._player_side == 'white' else 1
            assert self._board.turn == (0 if self._player_side == 'white' else 1)

        self._current = self._save_state()

        # Check whether the students installed all fences or not.
        if problem_type == 2:
            self.check_state_difference(state, self._current)

            fences_left_player = self.number_of_fences_left(self.get_player_id())
            fences_left_other = self.number_of_fences_left(self.get_opponent_id())

            assert fences_left_player == 0, 'Some fences left as unused after the simulation!'
            assert (FENCES_MAX * 2 - (fences_left_other + fences_left_other)
                    == len(self._board.fence_center_grid.argwhere().tolist())), \
                'Number mismatch! The number of recorded fence usage != The number of installed fences'

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('State has been changed to: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))
            self._logger.debug('------- SIMULATION ENDS -------')

        # Update memory usage
        self._update_memory_usage()

        return deepcopy(self._current)

    def _unique_game_state_identifier(self) -> str:
        """
        Return the unique identifier for game states.
        If two states are having the same identifier, then the states can be treated as identical in this problem.

        :return: String of game identifier
        """

        return self._board.partial_FEN()

    def _save_state(self) -> dict:
        """
        Helper function for saving the current state representation as a python dictionary from the game board.

        :return: State representation of a game (in basic python objects)
        """

        return {
            'state_id': self._unique_game_state_identifier(),
            # Unique identifier for the game state. If this is the same, then the state will be equivalent.
            'player_id': self._player_side,  # The agent's Player ID
            'turn': self._board.turn,  # Currently playing Player's ID
            'board': {  # Information about the current board
                'fence_center': self._board.fence_center_grid.argwhere().tolist(),
                'horizontal_fences': self._board.horizontal_fence_grid.argwhere().tolist(),
                'vertical_fences': self._board.vertical_fence_grid.argwhere().tolist(),
            },
            'player': {
                p: {  # Information about the current player
                    'pawn': list(self._board.pawns[p].square.location),
                    'fences_left': self._board.fences_left[p]
                }
                for p in ['black', 'white']
            }
        }

    def _restore_state(self, state: dict):
        """
        Helper function to restore board state to given state representation.
        """
        # Clear everything
        self._board.fence_center_grid[:, :] = False
        self._board.horizontal_fence_grid[:, :] = False
        self._board.vertical_fence_grid[:, :] = False

        # Set players' current position
        for p in ['black', 'white']:
            self._board.pawns[p].move(self._board.get_square_or_none(*state['player'][p]['pawn']))

        # Recover route-related information (before building fences)
        for pawn in self._board.pawns.values():
            pawn.square.reset_neighbours()

        for pawn in self._board.pawns.values():
            self._board.update_neighbours(pawn.square)

        # Recover fences. Before recovery, give all the fences to the current player.
        current_player = self._board.current_player()
        self._board.fences_left[current_player] = FENCES_MAX * 2

        # Re-simulate fencing:
        horizontals = [tuple(place) for place in state['board']['horizontal_fences']]
        verticals = [tuple(place) for place in state['board']['vertical_fences']]
        for r, c in state['board']['fence_center']:
            if (r, c) in horizontals and (r, c + 1) in horizontals:
                act = BLOCK(player=current_player, orientation='horizontal', edge=(r, c))
            elif (r, c) in verticals and (r + 1, c) in verticals:
                act = BLOCK(player=current_player, orientation='vertical', edge=(r, c))
            else:
                raise ValueError(f'Fence center {r}, {c} is not in both horizontal and vertical fences')
            try:
                act(self)
            except GameOver:
                continue

        # Set players' information
        for p in ['black', 'white']:
            self._board.fences_left[p] = state['player'][p]['fences_left']

        self._player_side = state['player_id']
        self._board.turn = state['turn']


# Export only GameBoard and RESOURCES.
__all__ = ['GameBoard', 'IS_DEBUG', 'IS_RUN']
