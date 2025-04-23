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


def _execute(initial_state, cls, logger):
    process = psutil.Process(os.getpid())

    # Set up the given problem
    board = GameBoard()
    board._initialize()

    # Apply distance conditions
    distances = initial_state.pop('distances')
    board._vertical_turns = distances['vertical']
    board._horizontal_turns = distances['horizontal']

    # Copy the initial set of fences
    initial_fences = {k: v.copy() for k, v in initial_state['board'].items()}

    # Reset to specific state
    board.set_to_state(initial_state, is_initial=True)
    board._set_initial_fences(initial_fences)

    # For each agent, execute the same problem.
    a = cls(player=initial_state['player_id'])

    # Initialize board and log initial memory size
    board.reset_memory_usage()
    board.get_current_memory_usage()

    final_answer = None
    state = None
    failure = None
    peak_memory = initial_memory = 0

    # Start to search
    logger.info(f'Begin to search using {a.name} agent.')
    time_delta = 0
    try:
        def update_memory():
            nonlocal peak_memory, initial_memory
            try:
                current = process.memory_info().rss / MEGABYTES
                peak_memory = max(peak_memory, current)
                if initial_memory == 0:
                    initial_memory = peak_memory
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

        time_start = time()
        final_answer = a.local_search(board, time_limit=time_limit)
        assert isinstance(move, list) and len(move) == 3 and all(isinstance(x, BLOCK) for x in move),\
            'Solution should be a LIST of 3 BLOCKs.'

        time_end = time()
        time_delta = time_end - time_start

        stop_monitoring = True
        monitor_thread.join(timeout=1.0)

        if final_answer is None:
            failure = 'Algorithm failed to provide BLOCK() action within the time limit.'
    except:
        failure = traceback.format_exc()
    finally:
        stop_monitoring = True

    # Compute how much time passed
    time_delta = round(time_delta * 100) / 100
    search_call = board._heuristic_calls

    board_memory = board.get_max_memory_usage() / MEGABYTES
    memory_usage = round(max(board_memory, peak_memory - initial_memory) * 100) / 100

    if time_delta > HARD_TIME_LIMIT > 0:
        return Performance(
            failure=f'Time limit exceeded! {time_delta:.3f} seconds passed!',
            outcome=float('inf'),
            search=None,
            time=time_delta,
            memory=memory_usage,
            point=1 # Just give submission point
        )
    if memory_usage > HARD_MEMORY_LIMIT > 0:
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
            next_state = board.simulate_action(None, *final_answer, problem_type=2)

            # Now, use agent to find the shortest path of opponent.
            # TODO: I'll provide TA's answer for part I, after the submission.
            length = board.distance_to_goal(board.get_opponent_id())

            # Check whether the initial fences exists
            for key in ['fence_center', 'horizontal_fences', 'vertical_fences']:
                next_fences = {tuple(r) for r in next_state['board'][key]}
                init_fences = {tuple(r) for r in initial_fences[key]}

                assert next_fences.issuperset(init_fences), 'Initial fences are moved!'
        except (InvalidMove, InvalidFence):
            failure = traceback.format_exc()

    return Performance(
        failure=failure,
        outcome=length,
        search=search_call,
        time=time_delta,
        memory=memory_usage,
        point=1
    )



def execute_local_search(agent, initial_state: dict, logger: Logger, res_ta: Performance = None):
    res = _execute(initial_state.copy(), agent, logger)

    # Give points by the stage where the agent is.
    is_beating_ta_outcome = True
    is_beating_ta_time = False
    if res_ta is not None:
        # When TA agent exists, apply TA's result.
        is_beating_ta_outcome = res_ta.outcome <= res.outcome
        is_beating_ta_time = res_ta.search * 1.01 >= res.search   # Allow 1% errors

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