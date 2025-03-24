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


#여기
import sys
from util import print_board



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

# problem 1이므로 * Outcome = Length of found path (smaller = better).* SearchActs are not used in this challenge. 출력됨


def _nan_format(value, length=8, decimal=1):
    if value is None or np.isnan(value):
        return '-' * (length - decimal) + '.' + '-' * (decimal - 1)
            
# NaN 은 float 형태의 결측값(None)을 의미

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

    #results > agent를 모아둔 dictionary     keys > agent 이름
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
        
        #각 agent(r)의 failure, outcome, memory, timespent, search, point를 구한다.

        # Get last item
        last = results[agent][-1]

        # Name print option
        key_print = agent if len(agent) < 13 else agent[:9] + '...'
        # agent 이름이 13글자 미만이면 그대로 출력, 이상이면 ... 을 사용하여 요약

        print("여기야아아아") # agent당 각 값 출력
        print(f' {key_print:12s} | {failure} {outcome} {memory}MB {timespent}sec {search} | {points}')
        print(f'   +- lastRun | {"FAILURE " if last.failure is not None else " " * 8} '
              f'{_nan_format(last.outcome, 7, 2)} {_nan_format(last.memory, 7, 2)}MB '
              f'{_nan_format(last.time, 6, 2)}sec {_nan_format(last.search, 10, 2)} '
              f'| {_nan_format(last.point, 5, 3)}')



        # Write-down the failures
        with Path(f'./failure_{agent}.txt').open('w+t') as fp:
            fp.write('\n-----------------\n'.join([r.failure for r in results[agent] if r.failure is not None]))

        # failure 내용을 파일에 저장. join 함수는 list element를 하나의 string으로 합친다.

    print(NOTE[part]) # 실행할때 part를 1로 설정하니까 part 1 note가 출력된다.

def _read_result(res_queue, last_execution):
    """
    Read evaluation result from the queue.
    :param res_queue: Queue to read
    """
    while not res_queue.empty():
        agent_i, perf_i = res_queue.get()
        if agent_i not in last_execution or last_execution[agent_i] is None:
            last_execution[agent_i] = perf_i

        # last_execution 이라는 dictionary가 __main__ 에 있는데, agent의 performance를 저장하는 듯
     


def _execute(part, prob, agent, process_results, last_execution, **kwargs):
    """
    Execute an evaluation for an agent with given initial state.
    :param part: Challenge part number
    :param prob: Initial state for a problem
    :param agent: Agent
    :return: A process
    """

    #Process 는 멀티쓰레드 라이브러리에 있는 class임    traget > 함수 / args > parameter 같은데 evaluate_algorithm 이라는 함수가 어딨지..
    proc = Process(name=f'EvalProc', target=evaluate_algorithm, args=(agent, prob, part, process_results),
                   kwargs=kwargs, daemon=True)   # daemon = True > main process 종료 시 sub process 도 함께 종료
    proc.start()
    proc.agent = agent  # Make an agent tag for this process
    last_execution[agent] = None
    return proc   # process return


# Main function
if __name__ == '__main__':

    #여기
    IS_DEBUG = '--debug' in sys.argv
    logging.basicConfig(level=logging.DEBUG if IS_DEBUG else logging.INFO)


    # Definition of arguments, used when running this program
    argparser = ArgumentParser()   #command 창에서 argument를 입력 받을 수 있도록 하는 라이브러리
    argparser.set_defaults(debug=False)   # 아무런 입력을 받지 않았을 때
    argparser.add_argument('-p', '-part', '--part', type=int, choices=[1, 2, 3, 4],
                           help='Challenge Part ID: 1, 2, 3, 4. E.g., To check your code with Part III, use --part 3')
    argparser.add_argument('--debug', action='store_true', dest='debug',
                           help='Enable debug mode')
    args = argparser.parse_args()   #위에 있는 인자들을 입력으로 받는다.

    # Problem generator for the same execution
    prob_generator = GameBoard()
    # GameBord 객체


    # List of all agents
    all_agents = get_all_agents()
    #from agents.load import get_all_agents 소속. agent를 list로 모두 불러온다.

    # Set a random seed
    seed(42)

    # Performance measures
    performances = defaultdict(list)   #defauldict은 아직 존재하지 않는 key를 조회할 때 기본값을 생성해줌. 기본값이 list 이므로 []을 생성해줌
    last_execution = {}

    # Start evaluation process (using multi-processing)
    process_results = Queue(len(all_agents) * 2)
    process_count = max(cpu_count() - 2, 1)
    # all agents는 아까 get_all_agents()로 불러온 agent list. length는 agent 개수가 됨
    # Queue()는 multiprocessing.Queue 객체로, 프로세스 간 데이터를 주고받을 때 사용해.
    #  8코어 CPU에서는 cpu_count()가 8을 반환해. >> process 를 몇개 만들건지 설정하는게 process_count

    # Run multiple times
    #for t in range(GAMES):  #GAMES 는 몇 게임할건지. 10판으로 설정되어 있음
    for t in range(1):
        # Clear all previous results
        last_execution.clear()   #clear()는 list, dict, set의 모든 element를 제거함
        while not process_results.empty():
            process_results.get()   #Queue가 빌 때까지 값을 꺼냄 


        #여기서부터 다시
        # Generate new problem   prob_generator 는 GameBoard 객체
        prob_generator._initialize(start_with_random_fence=STARTING_FENCES[args.part])  #part1의 random fence는 4개씩이다.
        prob_spec = prob_generator.get_initial_state()  #prob_spec 은 현재 보드 상태를 할당 받음. 시작 위치 변경, fence 설치 까지 되잇는 상태
        logging.info(f'Trial {t} begins!') # 겜 시작

        # Add random information for Part 3.
        if args.part == 3:
            prob_spec['random_action_indices'] = [randint(0, 65536) for _ in range(4)]  #이건 part 3거 나중에 보자


        #여기
        all_agents_test = get_all_agents()
        print("여기다 !!! get state: ", prob_generator.get_player_id())
        sys.stdout.flush() 
        #print(prob_spec)

        

        # Execute agents
        processes = []  # 멀티 프로세스 목록인듯
        agents_to_run = all_agents.copy()  # agent 목록
        shuffle(agents_to_run) # agent 순서 섞기

        while agents_to_run or processes:
            # If there is a room for new execution, execute new thing.
            if agents_to_run and len(processes) < process_count:
                alg = agents_to_run.pop()  # 실행할 agent를 하나 꺼냄
                processes.append((_execute(args.part, prob_spec, alg, process_results, last_execution), time()))
            # excute 함수로 에이전트 실행 / processes 리스트에 추가
            #agents_to_run에 아직 실행할 에이전트가 남아 있고,
            # 현재 실행 중인 프로세스 개수(len(processes))가 최대 프로세스 개수(process_count)보다 적다면, 새로운 프로세스 실행



            new_proc_list = []
            for p, begin in processes:  #p는 process, begin은 time()
                if not p.is_alive():  # 실행 끝났으면 넘어가는 듯
                    continue

                # Print running info
                now = time()  # 현재 시간
                print(f'Running "{p.agent}" for {now - begin:4.0f}/{TIME_LIMIT} second(s).', end='\r')  #얼마동안 실행되고 있는지 출력인듯. 근데 안보이네

                # For each running process, check memory usage
                try:
                    p_mb = pu.Process(p.pid).memory_info().rss / MEGABYTES
                except pu.NoSuchProcess:
                    new_proc_list.append((p, begin))
                    p_mb = 0    #process들의 memory 사용량 가져오기

                # For each running process, check for timeout
                time_spent = now - begin #현재시간 - 시작 시간 == 총 작동 시간
                if time_spent > TIME_LIMIT:  #시간 제한보다 크면 종료
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
                    )    #종료 로그 띄우고 그때까지의 performance 저장
                elif p_mb > MEMORY_LIMIT:  # memory 제한을 조과해도 똑같이
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
                    new_proc_list.append((p, begin))  #아까 위에서 추가 되었을 텐데..

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
                performances[agent_i].append(last_execution[agent_i])  #위에서 선언된 defaultdict
            else:  # Last execution failed
                performances[agent_i].append(Performance(
                    failure='No execution data found!',
                    memory=-1,
                    time=0,
                    outcome=None,
                    search=None,
                    point=1  # executrion이 failed인 경우, 이 값들로 저장
                ))

        # Sort the results for each performance criteria and give ranks to agents
        _print_table(t, args.part, performances)
