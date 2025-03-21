# Package for logging your execution
import logging
# Package for runtime importing
from importlib import import_module
# Package for multiprocessing (evaluation will be done with multiprocessing)
from multiprocessing import Queue
# Package for writing exceptions
from traceback import format_exc

# Package for problem definitions
from board import GameBoard, IS_RUN, IS_DEBUG
# Package for supporting evaluation
from .util import Performance
from .part1 import execute_heuristic_search
from .part2 import execute_local_search
from .part3 import execute_belief_state_search
from .part4 import execute_adversarial_search



def evaluate_algorithm(agent_name, initial_state, problem_id,
                       result_queue: Queue, res_ta: Performance = None):
    """
    Run the evaluation for an agent.
    :param agent_name: Agent to be evaluated
    :param initial_state: Initial state for the test
    :param problem_id: Problem ID (1, 2, 3, or 4)
    :param result_queue: A multiprocessing Queue to return the execution result.
    :param res_ta: TA agent's result (Performance instance)
    """
    # Initialize logger
    if not IS_RUN:
        logging.basicConfig(level=logging.DEBUG if IS_DEBUG else logging.INFO,
                            format='%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s',
                            filename=f'execution-{agent_name}.log',
                            # Also, the output will be logged in 'execution-(agent).log' file.
                            filemode='w+',
                            force=True, encoding='UTF-8')  # The logging file will be overwritten.
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s',
                            force=True, encoding='UTF-8')
    logger = logging.getLogger('Evaluate')

    # Initialize an agent
    try:
        logger.info(f'Loading {agent_name} agent to memory...')
        module = import_module(f'agents.{agent_name}')
        agent = module.Agent
    except Exception as e:
        # When agent loading fails, send the failure log to main process.
        failure = format_exc()
        logger.error('Loading failed!', exc_info=e)
        result_queue.put(
            (agent_name,
             Performance(failure, outcome=None, time=None, search=None, memory=None))
        )
        return

    # Execute algorithm
    if problem_id == 1:
        performance = execute_heuristic_search(agent, initial_state, logger, res_ta=res_ta)
    elif problem_id == 2:
        performance = execute_local_search(agent, initial_state, logger, res_ta=res_ta)
    elif problem_id == 3:
        performance = execute_belief_state_search(agent, initial_state, logger, res_ta=res_ta)
    else:
        performance = execute_adversarial_search(agent, initial_state, logger, res_ta=res_ta)

    if IS_DEBUG:
        logger.debug(f'Execution Result: {performance}.')
    result_queue.put((agent_name, performance))
