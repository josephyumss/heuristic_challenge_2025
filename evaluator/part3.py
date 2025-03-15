import traceback
from time import time
from logging import Logger
from typing import Literal

from pyquoridor.exceptions import InvalidMove, InvalidFence

from action import Action, MOVE, BLOCK
from board import GameBoard
from .util import Performance, MEGABYTES, load_ta_agent

HARD_TIME_LIMIT = 600
HARD_MEMORY_LIMIT = 1024


def execute_random_action(game: GameBoard, player: Literal['white', 'black'],
                          state: dict, seed: int):
    fence_actions = [BLOCK(player, *f)
                     for f in game.get_applicable_fences(player)]
    move_actions = [MOVE(player, m)
                    for m in game.get_applicable_moves(player)]

    actions = fence_actions + move_actions
    while len(actions) > 0:
        try:
            action = actions[seed % len(actions)]
            actions.remove(action)
            new_state = game.simulate_action(state, action)
            return new_state
        except (InvalidMove, InvalidFence):
            pass

    raise InvalidMove('No possible move left for random agent!')


def execute_belief_state_search(agent, initial_state: dict, logger: Logger):
    # Pop random agent action from the state
    action_seed = initial_state.pop('random_action_indices')

    # Set up the given problem
    board = GameBoard()
    board._initialize()

    # Load TA agent if exists
    agents = {'agent': agent}
    ta_agent = load_ta_agent(board)
    heuristic_agent = agent
    if ta_agent is not None:
        agents['ta'] = ta_agent
        heuristic_agent = ta_agent

    # For each agent, execute the same problem.
    results = {}
    for k, a in agents.items():
        # Initialize board and log initial memory size
        board.set_to_state(initial_state, is_initial=True)
        board.reset_memory_usage()
        board.get_current_memory_usage()

        solution = None
        failure = None

        # Start to search
        logger.info(f'Begin to search using {a.name} agent.')
        time_start = time()
        time_limit = time_start + HARD_TIME_LIMIT
        try:
            solution = a.belief_state_search(board, time_limit=time_limit)
            assert isinstance(solution, list),\
                'Solution should be a LIST of actions. The provided value is not a list.'
            assert all(isinstance(s, Action) for s in solution), \
                'Solution should be a list of ACTIONs. The provided list contains non-Action instances.'
            assert len(solution) == 4, 'Solution should be a LIST of 4 actions.'
        except:
            failure = traceback.format_exc()

        # Compute how much time passed
        time_end = time()
        time_delta = int(max(0, int(time_end - time_start)))

        # Compute how much memory used during search.
        memory_usage = int(max(0, board.get_max_memory_usage()) / MEGABYTES)

        if k == 'agent' and time_delta > HARD_TIME_LIMIT > 0:
            return Performance(
                failure=f'Time limit exceeded! {time_delta} seconds passed!',
                outcome=float('inf'),
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1  # Just give submission point
            )
        if k == 'agent' and memory_usage > HARD_MEMORY_LIMIT > 0:
            return Performance(
                failure=f'Memory limit exceeded! {memory_usage} MB used!',
                outcome=float('inf'),
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1  # Just give submission point
            )

        # Now, evaluate the solution
        length = float('inf')
        point = 1
        if solution is not None:
            try:
                state = initial_state
                player_side = board.get_player_id()

                for i in range(4):
                    if player_side == 'white':
                        # Do white first.
                        state = board.simulate_action(state, solution[i])
                        # Do random action
                        state = execute_random_action(board, 'black',
                                                      state, seed=action_seed[i])
                    else:
                        # Do white first with random action.
                        state = execute_random_action(board, 'white',
                                                      state, seed=action_seed[i])
                        # Do black.
                        state = board.simulate_action(state, solution[i])

                # Now, use agent to find the shortest path of opponent.
                # TODO: I'll provide TA's answer for part I, after the submission.
                board._player_side = 'white' if player_side == 'black' else 'black'
                route = heuristic_agent.heuristic_search(board)

                length = len(route)
            except (InvalidMove, InvalidFence):
                failure = traceback.format_exc()

        results[k] = Performance(
            failure=failure,
            outcome=length,
            search=None,
            time=time_delta,
            memory=memory_usage,
            point=1
        )

    # Give points by the stage where the agent is.
    res = results['agent']

    is_beating_ta_outcome = False
    if 'ta' in results:
        # When TA agent exists, apply TA's result.
        is_beating_ta_outcome = results['ta'].outcome <= res.outcome

    is_basic_stage = res.failure is None
    is_intermediate_stage = is_basic_stage and (res.time <= 5)
    is_advanced_stage = is_intermediate_stage and is_beating_ta_outcome
    # TA computation time will be measured on online system.

    return Performance(
        failure=res.failure,
        outcome=res.outcome,
        search=res.search,
        time=res.time,
        memory=res.memory,
        point=1 + int(is_basic_stage) + int(is_intermediate_stage) + int(is_advanced_stage) * 2
    )
