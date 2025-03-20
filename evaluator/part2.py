import traceback
from time import time
from logging import Logger
import psutil
import os

from pyquoridor.exceptions import InvalidMove, InvalidFence

from action import Action, MOVE, BLOCK
from board import GameBoard
from .util import Performance, MEGABYTES, load_ta_agent

HARD_TIME_LIMIT = 300
HARD_MEMORY_LIMIT = 10


def execute_local_search(agent, initial_state: dict, logger: Logger):
    # Set up the given problem
    board = GameBoard()
    board._initialize()
    board.set_to_state(initial_state, is_initial=True)

    # Load TA agent if exists
    agents = {'agent': agent}
    ta_agent = load_ta_agent(board)
    heuristic_agent = agent
    if ta_agent is not None:
        agents['ta'] = ta_agent
        heuristic_agent = ta_agent

    process = psutil.Process(os.getpid())

    # For each agent, execute the same problem.
    results = {}
    for k in ['ta', 'agent']:
        a = agents[k]
        initial_memory = process.memory_info().rss / MEGABYTES

        # Initialize board and log initial memory size
        board.set_to_state(initial_state, is_initial=True)
        board.reset_memory_usage()
        board.get_current_memory_usage()

        search_call = 0
        final_answer = None
        state = None
        failure = None
        peak_memory = initial_memory

        # Start to search
        logger.info(f'Begin to search using {a.name} agent.')
        time_start = time()
        time_limit = time_start + HARD_TIME_LIMIT
        try:
            def update_memory():
                nonlocal peak_memory
                try:
                    current = process.memory_info().rss / MEGABYTES
                    peak_memory = max(peak_memory, current)
                except:
                    pass

            import threading
            stop_monitoring = False

            def memory_monitor():
                while not stop_monitoring:
                    update_memory()
                    import time
                    time.sleep(0.1)

            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()

            while time() < time_limit:
                move = a.local_search(board, time_limit=time_limit)
                search_call += 1
                assert isinstance(move, MOVE) or (isinstance(move, list) and len(move) == 3
                                                  and all(isinstance(x, BLOCK) for x in move)),\
                    'Solution should be either MOVE or a LIST of 3 BLOCKs.'

                if isinstance(move, MOVE):
                    state = board.simulate_action(state, move, problem_type=1)
                else:
                    final_answer = move
                    break

            stop_monitoring = True
            monitor_thread.join(timeout=1.0)

            if final_answer is None:
                failure = 'Algorithm failed to provide BLOCK() action within the time limit.'
        except:
            failure = traceback.format_exc()
        finally:
            stop_monitoring = True

        # Compute how much time passed
        time_end = time()
        time_delta = round((time_end - time_start) * 100) / 100

        board_memory = board.get_max_memory_usage() / MEGABYTES
        memory_usage = round(max(board_memory, peak_memory - initial_memory) * 100) / 100

        if k == 'agent' and time_delta > HARD_TIME_LIMIT > 0:
            return Performance(
                failure=f'Time limit exceeded! {time_delta:.3f} seconds passed!',
                outcome=float('inf'),
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1 # Just give submission point
            )
        if k == 'agent' and memory_usage > HARD_MEMORY_LIMIT > 0:
            return Performance(
                failure=f'Memory limit exceeded! {memory_usage:.2f} MB used!',
                outcome=float('inf'),
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1 # Just give submission point
            )

        # Now, evaluate the solution
        length = None
        if final_answer is not None:
            try:
                board.set_to_state(initial_state, is_initial=True)  # Reset to initial state
                board.simulate_action(None, *final_answer, problem_type=1)

                # Now, use agent to find the shortest path of opponent.
                # TODO: I'll provide TA's answer for part I, after the submission.
                board._player_side = 'white' if board._player_side == 'black' else 'black'
                route = heuristic_agent.local_search(board)

                length = len(route)
            except (InvalidMove, InvalidFence):
                failure = traceback.format_exc()

        results[k] = Performance(
            failure=failure,
            outcome=length,
            search=search_call,
            time=time_delta,
            memory=memory_usage,
            point=1
        )

    # Give points by the stage where the agent is.
    res = results['agent']

    is_beating_ta_outcome = True
    is_beating_ta_time = False
    if 'ta' in results:
        # When TA agent exists, apply TA's result.
        is_beating_ta_outcome = results['ta'].outcome <= res.outcome
        is_beating_ta_time = results['ta'].search >= res.search

    is_basic_stage = (res.failure is None) and is_beating_ta_outcome
    is_intermediate_stage = is_basic_stage and (res.time <= 10)
    is_advanced_stage = is_intermediate_stage and (res.memory <= 1)
    is_challenge_stage = is_advanced_stage and is_beating_ta_time
    # TA computation time will be measured on online system.

    return Performance(
        failure=res.failure,
        outcome=res.outcome,
        search=res.search,
        time=res.time,
        memory=res.memory,
        point=1 + int(is_basic_stage) + int(is_intermediate_stage) + int(is_advanced_stage) + int(is_challenge_stage)
    )
