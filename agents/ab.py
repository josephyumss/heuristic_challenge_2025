from pathlib import Path
from random import choice
from typing import List, Literal, Union
from queue import PriorityQueue
from action import MOVE, BLOCK, Action
from board import GameBoard


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
        return []

    def local_search(self, board: GameBoard, time_limit: float) -> Union[MOVE, List[BLOCK]]:
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
        :return: The next MOVE or list of three BLOCKs.
            That is, you should either return MOVE() action or [BLOCK(), BLOCK(), BLOCK()].
        """
        return [
            BLOCK(player=self.player, edge=(1, 1), orientation='vertical'),
            BLOCK(player=self.player, edge=(2, 2), orientation='vertical'),
            BLOCK(player=self.player, edge=(3, 3), orientation='vertical')
        ]

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
        return [BLOCK(self.player, edge=(1,1), orientation='horizontal'),
                BLOCK(self.player, edge=(7,7), orientation='horizontal'),
                BLOCK(self.player, edge=(3,3), orientation='horizontal'),
                BLOCK(self.player, edge=(6,6), orientation='horizontal'),]

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
        from heapq import heappop, heappush
        import time

        start_time = time.time()
        TIME_BUFFER = 1.0  # 1초 남기고 종료

        player = board.current_player()
        opponent = 'white' if player == 'black' else 'black'
        max_depth = 2

        def shortest_path_length(position, goal_rows):
            visited = set()
            frontier = [(0, position)]
            while frontier:
                cost, current = heappop(frontier)
                if current[0] in goal_rows:
                    return cost
                if current in visited:
                    continue
                visited.add(current)
                board._board.pawns[player].move(board._board.get_square_or_none(*current))
                for square in board._board.valid_pawn_moves(player, check_winner=False):
                    neighbor = (square.row, square.col)
                    if neighbor not in visited:
                        heappush(frontier, (cost + 1, neighbor))
            return 99

        def evaluate(state):
            try:
                my_pos = state['player'][player]['pawn']
                opp_pos = state['player'][opponent]['pawn']
                my_goal = [8] if player == 'white' else [0]
                opp_goal = [8] if opponent == 'white' else [0]

                board.set_to_state(state)
                board._board.pawns[player].move(board._board.get_square_or_none(*my_pos))
                my_len = shortest_path_length(tuple(my_pos), my_goal)

                board.set_to_state(state)
                board._board.pawns[opponent].move(board._board.get_square_or_none(*opp_pos))
                opp_len = shortest_path_length(tuple(opp_pos), opp_goal)

                return opp_len - my_len
            except:
                return 0

        def minimax(state, depth, alpha, beta, maximizing):
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
                        eval, _ = minimax(new_state, depth - 1, alpha, beta, False)
                        if eval > max_eval:
                            max_eval = eval
                            best_action = act
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
                    except:
                        continue
                return max_eval, best_action
            else:
                min_eval = float('inf')
                for act in actions:
                    try:
                        new_state = board.simulate_action(state, act)
                        eval, _ = minimax(new_state, depth - 1, alpha, beta, True)
                        if eval < min_eval:
                            min_eval = eval
                            best_action = act
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
                    except:
                        continue
                return min_eval, best_action

        try:
            value, action = minimax(board.get_state(), max_depth, float('-inf'), float('inf'), True)
            if action:
                return action
        except TimeoutError:
            pass
        except:
            pass

        fallback = board.get_applicable_moves(player)
        if fallback:
            return MOVE(player, fallback[0])
        return MOVE(player, board.get_position(player))