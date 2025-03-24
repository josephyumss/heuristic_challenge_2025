# Logging method for board execution
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
from pyquoridor.exceptions import GameOver, InvalidFence  # InvalidFence 추가
from pyquoridor.square import MAX_COL, MAX_ROW

# Import action specifications
from action import Action, BLOCK
# Import some utilities
from util import print_board

#: True if the program run with 'DEBUG' environment variable.
IS_DEBUG = '--debug' in sys.argv
IS_RUN = 'fixed_evaluation' in sys.argv[0]


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

    #여기
    _logger.propagate = True

    #: [PRIVATE] Memory usage tracker
    _process_info = None
    #: [PRIVATE] Fields for computing maximum memory usage. Don't access this directly in your agent code!
    _init_memory = 0
    _max_memory = 0
    #: [PRIVATE] Random seed generator
    _rng = random.Random(2938)
    #: [PRIVATE] Remaining amount of fences unused.
    _fence_count = {'black': 10, 'white': 10}
    

    #evaluate __main__에서 prob_generator 가 여기서 init됨
    def _initialize(self, start_with_random_fence: int = 0):   #시작할 때 랜덤으로 설치한 fence의 갯수
        """
        Initialize the board for evaluation. ONLY for evaluation purposes.
        [WARN] Don't access this method in your agent code.
        """
        # Initialize process tracker
        self._process_info = PUInfo(os.getpid())
        # 현재 작동중인 각 process의 process id를 가져옴


        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Initializing a new game board...')
        
        # Initialize a new game board
        self._board = Board()  #from pyquoridor.board import Board 에서 가져오는 Board 객체 생성
        self._fence_count = {'black': 10, 'white': 10}  #fence count도 다시 initialize

        self._vertical_turns = [[self._rng.randint(1, 5) for _ in range(9)] for _ in range(8)]
        self._horizontal_turns = [[self._rng.randint(1, 5) for _ in range(8)] for _ in range(9)]
        #_rng.randint(1, 5)는 1부터 5 사이의 랜덤한 값을 생성하는 함수. 이동할 때마다 1~5 사이의 턴이 소모되도록 설정
        #보드 상에서 vertical move는 총 row 개수 -1 만큼 가능 따라서 for문이 저렇게 9,8 8,9 되는 거임 전체 사이즈는 9x9


        # Initialize board renderer for debugging purposes
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Rendered board: \n' + self._unique_game_state_identifier())   #게임 state의 id 같은 것 //E1E9 라고 출력된 예시
            self._logger.debug('\n' + print_board(self._board))   #print board로 보드 출력

        # Pick a starting point randomly.
        self._player_side = random.choice(['black', 'white'])   #내 player가 black 인지 white인지 랜덤으로 선택됨
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'You\'re player {self._player_side}')  # You're player black 라고 출력됨

        for p in ['black', 'white']:
            row = self._board.pawns[p].square.row    #square 는 격자 보드판의 각 칸인듯
            col = random_integer(0, MAX_COL - 1)    # col은 랜덤 설정. 시작 라인에서 랜덤한 좌우 위치로 시작하는 듯
            self._board.pawns[p].move(self._board.get_square_or_none(row, col))
            #self.pawns = {'white': Pawn(square=self[(white_init_row, white_init_col)], color='white') 라고 나와있음. 
            # pawns는 dict임 key : white/black ,  value : Pawn객체(square로 위치 나와있음, color(흰/검))
            #initial 위치에 있는 pawn의 row만 가져다 쓰고, col은 랜덤으로 재설정
            #move를 통해 특정 square로 이동시킴. Gameboard 객체의 field인 보드 객체에 square = self[(row, col)] 반환하여 해당 square로 이동시킴

        

        # Update position information with a new starting point
        for pawn in self._board.pawns.values():   #각 pawn 객체
            pawn.square.reset_neighbours()  #neighbours 는 위 양옆 아래 칸의 set  새로운 위치로 이동해서 다시 neighbours를 할당함

        for pawn in self._board.pawns.values():
            self._board.update_neighbours(pawn.square)  #보드 상에서 update 하여 이동가능한 neighbors를 찾는 듯. 보니까 상대말이 있는 칸을 찾는듯
            #return squares of neighbour squares occupied by pawns 라고 함

        # Set random fences   위에 gameboard initial 당시, start_with_random_fence 를 인자로 받음. 5 미만으로 설정해야 하는 듯
        assert start_with_random_fence < 5, 'Do not use start_with_random_fence >= 5'
        for _ in range(start_with_random_fence):   # 
            for p in self._board.pawns.keys():   #p는 white / black
                fences = self.get_applicable_fences(p) # 가능한 fence 설치 위치 튜플들의 리스트
                while fences:
                    fence, orientation = self._rng.choice(fences)  #fence에서 무작위로 뽑음 ((0, 0), 'horizontal') 이런 식이니 위치가 fence, 방향이 orientation
                    try:
                        BLOCK(p, fence, orientation)(self)  #BLOCK 은 Action class 보아하니 white/black 말, 위치, 방향을 인자로 받아서 막는 듯
                        break
                    except InvalidFence:
                        fences.remove((fence, orientation)) #가능한 위치가 아니면 리스트에서 제거
                        continue
                    except:
                        raise

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('After moving initial position: \n' + self._unique_game_state_identifier()) #다시 게임 state id 호출
            self._logger.debug('\n' + print_board(self._board)) # 보드 설치 후 print

        # Store initial state representation
        self._initial = self._save_state()  #현재 게임 상태 저장
        self._current = deepcopy(self._initial)  #shallow copy >> 원본 객체의 주소를 복사 > 같이 바뀜 / deep copy >> 객체를 복제 생성 느낌. 서로 별개

        # Update memory usage
        self._update_memory_usage()  #memory 다시 update함



    
    def get_move_turns(self, current_pos: tuple, next_pos: tuple) -> int:
        """Return the number of turns required to move between adjacent positions"""
        row1, col1 = current_pos
        row2, col2 = next_pos
        if col1 == col2:
            min_row = min(row1, row2)
            return self._vertical_turns[min_row][col1]
        elif row1 == row2:
            min_col = min(col1, col2)
            return self._horizontal_turns[row1][min_col]
        else:
            return float('inf')  # 사실상 불가능한 움직임이니까 inf 반환
    
    def print_turns(self):   # 안쓰이는듯
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
        
    def reset_memory_usage(self):  #이것도 안쓰이는 듯
        """
        Reset memory usage
        """
        self._init_memory = 0
        self._max_memory = 0
        self._update_memory_usage()

    def set_to_state(self, specific_state=None, is_initial: bool = False):
        """
        Restore the board to the initial state for repeated evaluation.
        :param specific_state: A state representation which the board reset to
        :param is_initial: True if this is an initial state to begin evaluation
        """
        assert specific_state is not None or not is_initial
        if specific_state is None:
            specific_state = self._initial   #특정 state로 복구하는거 같은데, Default의 경우, initial state로 복구함
                                            #근데 위에 assert때문에 사실상 실행 안됨
             
        if is_initial:
            self._initial = specific_state
            self._current = deepcopy(self._initial)
            self._rng.seed(hash(self._initial['state_id']))  # Use state_id as hash seed.
            # 이것도 실행안되는듯

        # Restore the board to the given state.
        self._restore_state(specific_state)  #아래에 있음

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('State has been set as follows: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))

    def is_game_end(self):
        """
        Check whether the given state indicate the end of the game

        :return: True if the game ends at the given state
        """
        is_game_end = self._board.game_finished()   # white면 max row -1 / black 이면 min row
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'Querying whether the game ends in this state... Answer = {is_game_end}')
        return is_game_end

    def get_state(self) -> dict:   #중요
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
        return deepcopy(self._current)  # 아까 처음 initial stat deepcopy한 애


    
    def get_initial_state(self) -> dict:
        """
        Get the initial board state
        :return: A copy of the initial board state dictionary
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying initial state...')  #로그 띄우고

        # Check whether the game has been initialized or not.
        assert self._initial is not None, 'The board should be initialized. Did you run the evaluation code properly?'
        # Return the initial state representation as a copy.
        return deepcopy(self._initial)   # 아까 위에서 저장한 white/black 시작 위치 변경, fence 설치 상태의 initial state

    def get_player_id(self) -> Literal['black', 'white']:
        """
        Return the player name.
        """
        return self._player_side  # black/white player name을 return함 string

    def get_opponent_id(self) -> Literal['black', 'white']:
        """
        Return the opponent name.
        """
        return 'black' if self._player_side == 'white' else 'white'

    def get_applicable_moves(self, player: Literal['black', 'white'] = None) -> List[Tuple[int, int]]:
        """
        Get the list of applicable roads
        :param player: Player name. black or white. (You can ask your player ID by calling get_player_index())   # 중요
        :return: A copy of the list of applicable move coordinates.
            (List of Tuple[int, int].)
        """
        if IS_DEBUG:  # Logging for debug
            #self._logger.debug('Querying applicable move directions...')
            self._logger.debug('Querying applicable move directions...<Action>')  #fence랑 로그가 똑같아서 이걸로 잠깐 바꿔줌

        # Read all applicable positions
        player = self._board.current_player() if player is None else player
        applicable_positions = sorted([
            (square.row, square.col)
            for square in self._board.valid_pawn_moves(player, check_winner=False)
        ])    #valid_pawn_moves 는 {s: self.BFS_player(player, init_square=s)[1] 를 return함
              # s in pawn.square.neighbours  s는 neighbour square임  > 이게 for문의 square임
              # BFS_player는 return can_reach, L 를 가지는데, can_reach는 boolean, L은 path length 인 듯
              # 근데 이러면 가능한 무브 하나가 나오는게 아닌거같은데..


        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            #self._logger.debug(f'List of applicable move positions: {applicable_positions}')
            pass

        # Return applicable positions as list of tuples.
        return applicable_positions  #위에서 구한 list 반환   list [(row,col),(row,col)] 형태임

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
        if self._fence_count[player] == 0:
            if IS_DEBUG:  # Logging for debug
                self._logger.debug(f'{player} used all fences.')
            return []
        # 남은 fence 개수가 0이면 걍 로그 출력

        applicable_fences = []
        for r in range(MAX_ROW - 1):
            for c in range(MAX_COL - 1):
                # Pass positions whose center of a grid.
                if self._board.fence_center_grid[r, c]:
                    continue
                # board 객체는 fence_center_grid 라는 field가 있는데, self.fence_center_grid = Grid(max_row, max_col) 이렇게 있음
                # 이에 해당하는 row, col은 건너뛰는듯. 뭘 건너뛰는건지는 모르겠지만

                if not self._board.fence_exists(r, c, 'h'):
                    applicable_fences.append(((r, c), 'horizontal'))      # fence가 없으면 applicable_fence에 추가함
                if not self._board.fence_exists(r, c, 'v'):
                    applicable_fences.append(((r, c), 'vertical'))

        applicable_fences = sorted(applicable_fences)    #sort 하여 list 로 가짐

        # Update memory usage
        self._update_memory_usage()  # memory 관리하는 코드인가? ㅇㅇ update는 최대 메모리 기록을 update함

        if IS_DEBUG:  # Logging for debug
            #self._logger.debug(f'List of applicable fence positions: {applicable_fences}')  #가능한 fence position이 쭉 나옴
            pass
        # Return applicable positions as list of tuples.
        return applicable_fences  #각 element는 ((0, 0), 'horizontal') 이런 식의 튜플임

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
        # 현재 process의 memory 사용량 반환

    def get_max_memory_usage(self):
        """
        :return: Maximum memory usage for the process having this board
        """
        return max(0, self._max_memory - self._init_memory) 
    # max memory 반환

    def _update_memory_usage(self):
        """
        [PRIVATE] updating maximum memory usage
        """
        if self._max_memory >= 0:
            self._max_memory = max(self._max_memory, self.get_current_memory_usage())
            # 최대 메모리 사용량을 갱신해줌

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

        # state 를 지정해주지 않으면, initial state에서 움직임

        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'------- SIMULATION START: {actions} -------')
            self._logger.debug('checkcheckcheck')
        # Restore to the given state
        self.set_to_state(state)  #restore_state 사용하는데 아래에 정의되어 있음

        # 입력받은 action들
        for act in actions:  # For each actions in the variable arguments,
            # Run actions through calling each action object. If error occurs, raise as it is (except for GameOver)
            self._logger.debug(f'for문 들어왔다. problem type : {problem_type}')
            
            try:
                # For challenge I and II, force the current player to agents
                if problem_type < 3:  #challenge 1, 2가 해당
                    self._logger.debug('problem type < 3')
                    self._board.turn = 0 if self._player_side == 'white' else 1

                act(self)  #Action.py를 봐야할 듯..인데 모르겟네
                self._logger.debug(f'turn: {self._board.turn} -------')
            except GameOver:
                break
            

            # Break the loop if the game ends within executing actions.
            if self.is_game_end():
                self._logger.debug('골들어간거 board로 들어갔네')
                break

        # Copy the current state to return  challenge 1, 2가 해당
        if problem_type < 3:
            self._board.turn = 0 if self._player_side == 'white' else 1
            assert self._board.turn == (0 if self._player_side == 'white' else 1)

        self._current = self._save_state()  #현재 정보 저장

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('State has been changed to: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))
            self._logger.debug('------- SIMULATION ENDS -------')   #log로 보드 출력

        # Update memory usage
        self._update_memory_usage()

        return deepcopy(self._current) #보드 state를 반환함. simulate니까 내가 이걸 따로 저장해야하는 듯?

    def _unique_game_state_identifier(self) -> str:
        """
        Return the unique identifier for game states.
        If two states are having the same identifier, then the states can be treated as identical in this problem.

        :return: String of game identifier
        """

        return self._board.partial_FEN()   # board state id 반환

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
                    'pawn': list(self._board.pawns[p].square.location),  # pawn 위치? current state에 이게 저장되어 있는거니까
                     # current_state['player'][player_id]['pawn']  >> tuple로 위치 나올 거 같은데  중요
                    'fences_left': self._fence_count[p]
                }
                for p in ['black', 'white']
            }
        }

    def _restore_state(self, state: dict):
        """
        Helper function to restore board state to given state representation.
        """
        # Clear everything
        self._board.fence_center_grid.grid[:, :] = False
        self._board.horizontal_fence_grid[:, :] = False
        self._board.vertical_fence_grid[:, :] = False

        # Set players first
        for p in ['black', 'white']:
            self._board.pawns[p].move(self._board.get_square_or_none(*state['player'][p]['pawn']))
            self._fence_count[p] = state['player'][p]['fences_left']

        # Recover fences
        for r, c in state['board']['fence_center']:
            self._board.fence_center_grid[r, c] = True
        for r, c in state['board']['horizontal_fences']:
            self._board.horizontal_fence_grid[r, c] = True
        for r, c in state['board']['vertical_fences']:
            self._board.vertical_fence_grid[r, c] = True

        # Recover route-related information
        for pawn in self._board.pawns.values():
            pawn.square.reset_neighbours()

        for pawn in self._board.pawns.values():
            self._board.update_neighbours(pawn.square)

        self._player_side = state['player_id']
        self._board.turn = state['turn']

        # 특정 state로 바꾼다는거같음

# Export only GameBoard and RESOURCES.
__all__ = ['GameBoard', 'IS_DEBUG', 'IS_RUN']
