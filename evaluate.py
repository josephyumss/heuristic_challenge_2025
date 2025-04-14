# Parser for arguments
from argparse import ArgumentParser
# A dictionary class which can set the default value
from collections import defaultdict
# Package for multiprocessing (evaluation will be done with multiprocessing)
from multiprocessing import cpu_count, Process
# Package for file handling
from pathlib import Path
# Package for randomness and seed control
from random import seed, shuffle, randint
# Package for time management
from time import time, sleep
# Package for typing
from typing import List, Dict

# Package for numeric actions
import numpy as np
# Package for measuring process utilization (memory)
import psutil as pu

# Function for loading your agents
from agents.load import get_all_agents
# Package for problem definitions
from evaluator import *
from evaluator.util import MEGABYTES

#: The number of games to run the evaluation
GAMES = 10
#: Limit of execution. (60 minutes)
TIME_LIMIT = 3600
MEMORY_LIMIT = 4096
#: Number of starting fences for each problem.
STARTING_FENCES = {
    1: 4,
    2: 4,
    3: 2,
    4: 0
}
#: Note for metrics
NOTE = {
    1: '                    * Outcome = Length of found path (smaller = better).\n'
       '                            * SearchActs are not used in this challenge.\n',
    2: '      * Outcome = Length of opponent path after BLOCK (larger = better).\n'
       '                  * SearchActs = Number of search calls before decision.\n',
    3: '    * Outcome = Length of opponent path after 4 turns (larger = better).\n'
       '                            * SearchActs are not used in this challenge.\n',
    4: '                                                   * Outcome = Win Rate.\n'
       '                            * SearchActs are not used in this challenge.\n',
}


def _nan_format(value, length=8, decimal=1):
    if value is None or np.isnan(value):
        return '-' * (length - decimal) + '.' + '-' * (decimal - 1)

    return (f'%{length}.{decimal}f') % (value)


def _nan_mean_string(items, length=8, decimal=1):
    if len(items) == 0:
        return _nan_format(None, length, decimal)

    mean = sum(items) / len(items)
    return _nan_format(mean, length, decimal)


def _print_table(trial, part, results: Dict[str, List[Performance]]):
    """
    Helper function for printing rank table
    :param trial: Game trial number
    :param part: Challenge part number
    """

    # Print header
    print('-' * 72)
    print(f'\nCurrent game trial: #{trial}')
    print(f' AgentName    | FailRate Outcome MemoryUsg TimeSpent SearchActs | Score ')
    print('=' * 14 + '|' + '=' * 49 + '|' + '=' * 7)

    for agent in sorted(results.keys()):
        # Compute mean score
        failure = _nan_mean_string([int(r.failure is not None) for r in results[agent]],
                                   length=8, decimal=4)
        outcome = _nan_mean_string([int(r.failure is None) * r.outcome
                                    for r in results[agent] if r.outcome is not None],
                                   length=7, decimal=2)
        memory = _nan_mean_string([r.memory for r in results[agent] if r.memory is not None and r.memory >= 0],
                                  length=7, decimal=2)
        timespent = _nan_mean_string([r.time for r in results[agent]],
                                     length=6, decimal=2)
        search = _nan_mean_string([r.search for r in results[agent] if r.search is not None],
                                  length=10, decimal=2)
        points = _nan_mean_string([r.point for r in results[agent]],
                                  length=5, decimal=3)

        # Get last item
        last = results[agent][-1]

        # Name print option
        key_print = agent if len(agent) < 13 else agent[:9] + '...'

        print(f' {key_print:12s} | {failure} {outcome} {memory}MB {timespent}sec {search} | {points}')
        print(f'   +- lastRun | {"FAILURE " if last.failure is not None else " " * 8} '
              f'{_nan_format(last.outcome, 7, 2)} {_nan_format(last.memory, 7, 2)}MB '
              f'{_nan_format(last.time, 6, 2)}sec {_nan_format(last.search, 10, 2)} '
              f'| {_nan_format(last.point, 5, 3)}')

        # Write-down the failures
        with Path(f'./failure_{agent}.txt').open('w+t') as fp:
            fp.write('\n-----------------\n'.join([r.failure for r in results[agent] if r.failure is not None]))

    print(NOTE[part])

def _read_result(res_queue, last_execution):
    """
    Read evaluation result from the queue.
    :param res_queue: Queue to read
    """
    while not res_queue.empty():
        agent_i, perf_i = res_queue.get()
        if agent_i not in last_execution or last_execution[agent_i] is None:
            last_execution[agent_i] = perf_i


def _execute(part, prob, agent, process_results, last_execution, **kwargs):
    """
    Execute an evaluation for an agent with given initial state.
    :param part: Challenge part number
    :param prob: Initial state for a problem
    :param agent: Agent
    :return: A process
    """
    proc = Process(name=f'EvalProc', target=evaluate_algorithm, args=(agent, prob, part, process_results),
                   kwargs=kwargs, daemon=True)
    proc.start()
    proc.agent = agent  # Make an agent tag for this process
    last_execution[agent] = None
    return proc


# Main function
if __name__ == '__main__':
    # Definition of arguments, used when running this program
    argparser = ArgumentParser()
    argparser.set_defaults(debug=False)
    argparser.add_argument('-p', '-part', '--part', type=int, choices=[1, 2, 3, 4],
                           help='Challenge Part ID: 1, 2, 3, 4. E.g., To check your code with Part III, use --part 3')
    argparser.add_argument('--debug', action='store_true', dest='debug',
                           help='Enable debug mode')
    args = argparser.parse_args()

    # Problem generator for the same execution
    prob_generator = GameBoard()
    # List of all agents
    all_agents = get_all_agents()
    # Set a random seed
    seed(42)

    # Performance measures
    performances = defaultdict(list)
    last_execution = {}

    # Start evaluation process (using multi-processing)
    process_results = Queue(len(all_agents) * 2)
    process_count = max(cpu_count() - 2, 1)

    # Run multiple times
    for t in range(GAMES):
        # Clear all previous results
        last_execution.clear()
        while not process_results.empty():
            process_results.get()

        # Generate new problem
        prob_generator._initialize(start_with_random_fence=STARTING_FENCES[args.part])
        prob_spec = prob_generator.get_initial_state()
        # Apply the same distance condition across evaluations
        prob_spec['distances'] = {
            'horizontal': prob_generator._horizontal_turns,
            'vertical': prob_generator._vertical_turns
        }
        logging.info(f'Trial {t} begins!')

        # Add random information for Part 3.
        if args.part == 3:
            prob_spec['random_action_indices'] = [randint(0, 65536) for _ in range(4)]

        # Execute agents
        processes = []
        agents_to_run = all_agents.copy()
        shuffle(agents_to_run)

        while agents_to_run or processes:
            # If there is a room for new execution, execute new thing.
            if agents_to_run and len(processes) < process_count:
                alg = agents_to_run.pop()
                processes.append((_execute(args.part, prob_spec, alg, process_results, last_execution), time()))

            new_proc_list = []
            for p, begin in processes:
                if not p.is_alive():
                    continue

                # Print running info
                now = time()
                print(f'Running "{p.agent}" for {now - begin:4.0f}/{TIME_LIMIT} second(s).', end='\r')

                # For each running process, check memory usage
                try:
                    p_mb = pu.Process(p.pid).memory_info().rss / MEGABYTES
                except pu.NoSuchProcess:
                    new_proc_list.append((p, begin))
                    p_mb = 0

                # For each running process, check for timeout
                time_spent = now - begin
                if time_spent > TIME_LIMIT:
                    p.terminate()
                    logging.error(f'[TIMEOUT] {p.agent} / '
                                  f'Process is running more than {TIME_LIMIT} sec, from ts={begin}; now={time()}')
                    last_execution[p.agent] = Performance(
                        failure=f'Process is running more than {TIME_LIMIT} sec, from ts={begin}; now={time()}',
                        memory=p_mb,
                        point=1,
                        search=None,
                        time=time_spent,
                        outcome=None
                    )
                elif p_mb > MEMORY_LIMIT:
                    p.terminate()
                    logging.error(f'[MEM LIMIT] {p.agent} / '
                                  f'Process consumed memory more than {MEMORY_LIMIT}MB (used: {p_mb}MB)')
                    last_execution[p.agent] = Performance(
                        failure=f'Process consumed memory more than {MEMORY_LIMIT}MB (used: {p_mb}MB)',
                        memory=p_mb,
                        point=1,
                        search=None,
                        time=time_spent,
                        outcome=None
                    )
                else:
                    new_proc_list.append((p, begin))

            # Prepare for the next round
            processes = new_proc_list
            # Read result from queue
            _read_result(process_results, last_execution)

            # Wait for one seconds
            sleep(1)

        # Read results finally
        logging.info(f'Reading results at Trial {t}')
        _read_result(process_results, last_execution)

        # Merge last execution result to results
        for agent_i in all_agents:
            last = last_execution[agent_i]
            if last is not None:
                performances[agent_i].append(last_execution[agent_i])
            else:  # Last execution failed
                performances[agent_i].append(Performance(
                    failure='No execution data found!',
                    memory=-1,
                    time=0,
                    outcome=None,
                    search=None,
                    point=1
                ))

        # Sort the results for each performance criteria and give ranks to agents
        _print_table(t, args.part, performances)