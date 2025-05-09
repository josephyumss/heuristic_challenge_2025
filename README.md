# CAU57456 Challenge: Algorithm for Quoridor

## Overview / 개관


This semester's programming challenge is to create an agent that plays the Quoridor game.

이번 학기의 프로그래밍 도전과제는 쿼리도 게임을 하는 에이전트를 만드는 것입니다.

The challenge consists of 4 problems: (1) Heuristic Search, (2) Local Search, (3) Belief-State Search, and (4) Adversarial Search.

도전과제는 4개의 문제로 구성되어 있습니다: (1) 휴리스틱 탐색, (2) 국소 탐색, (3) 믿음공간 탐색, (4) 적대적 탐색.

Based on this codebase, you need to write a program that solves the 4 parts, and if your performance meets the given criteria below, you will earn points.

이 코드베이스를 기준으로 여러분은 4개 부분을 해결하는 프로그램을 작성하고, 아래에 작성된 기준보다 더 높은 성능을 얻으면 됩니다.

The scoring criteria are categorized into Basic, Intermediate, Advanced, and Challenge levels, and your score will be determined based on whether your agent satisfied each level. You will receive 1 point for submitting working code, 2 points for satisfying the Basic-level criteria, 3 points for satisfying the Intermediate-level criteria, and 4 points for satisfying the Advanced-level criteria. You will earn 5 points for defeating the Challenge-level agent, and there are additional bonus points available. Bonus points are applied without exceeding the total score.

채점 기준은 **기초**, **중급**, **고급**, **도전** 단계로 구분되며, 어떤 단계의 기준을 만족했는지에 따라 점수가 결정됩니다. 동작하는 코드를 제출하였을 때 1점, 기초 에이전트를 이겼을 때 2점, 중급 에이전트를 이겼을 때에는 3점, 고급 에이전트를 이겼을 때에는 4점이 주어집니다. 도전 단계의 에이전트를 이겼을 때에는 5점이 주어지며, 별도의 가산점이 주어지는 항목들이 있습니다. 가산점은 점수 합계를 넘지 않는 선에서 적용됩니다.

Below, the detailed PEAS description for each problem is provided. Please write your code based on the PEAS description.

아래에 각 문제의 PEAS 상세정보가 주어져 있습니다. PEAS 상세정보를 바탕으로 여러분의 코드를 작성하세요.

(Are you curious about how to run the evaluation? Then, go to [RUN](./#RUN) section!)

(어떻게 평가를 실행하는지 궁금하다면, [RUN](./#RUN) 부분으로 넘어가세요!)

## PEAS description

### Performance measure (수행지표)

With one exception, everything follows the basic rules of Quoridor. Unlike the movement rules in Quoridor, in this challenge, the number of turns required to move between each cell varies. For example, moving up from a certain cell may take 3 turns, moving right may take 5 turns, moving left may take 1 turn, and moving down may take 4 turns. Different cells have different movement times. All other rules are the same as in Quoridor.

한 가지 사항을 제외하면, 모든 것은 기본적인 쿼리도의 규칙을 따릅니다. 쿼리도의 이동 규칙과 다르게, 이 도전과제 전체에서는 각 칸 사이의 이동에 소요되는 턴 수가 서로 다릅니다. 예를 들어, 어떤 칸에서 위로 갈 때는 3턴이 걸리고, 오른쪽으로 갈 때는 5턴, 왼쪽으로는 1턴, 아래로는 4턴이 걸릴 수 있습니다. 다른 칸에서는 다른 이동시간이 적용됩니다. 이 외의 모든 규칙은 쿼리도와 동일합니다.

The determination of "winning" is done through absolute evaluation. If any action that is not allowed in basic Quoridor is taken, it will result in disqualification. In addition to the actions that are not allowed in the game, there may be additional disqualification conditions depending on the assignment. In case of disqualification, only the base score of 1 point can be earned.

'이겼다'의 판단은 절대평가로 진행됩니다. 기본적인 쿼리도의 규칙을 따르며, 쿼리도에서 할 수 없는 행동을 한 경우에는 실격됩니다.
또한, 게임 내 불가능한 행동 외에도 과제에 따라 서로 다른 실격 조건이 추가됩니다. 실격패인 경우, 제출 기본 점수 1점만 얻을 수 있습니다.

All evaluations, except for Part 4, are based on the average score of 30 repeated trials. The initial position is randomly selected from one of the 10 starting cells.

Part 4를 제외한 모든 평가는, 30회 반복 시행의 점수 평균치를 기준으로 합니다. 초기 위치는 출발선 10칸 중 랜덤하게 정해집니다.

#### Part 1, Heuristic search

The goal of this problem is to find the shortest path from the current position to the victory point. Note that the victory point is not just one cell, but all the cells in the opponent’s starting line.

이 문제의 목표는 현 위치에서 승리지점까지의 최단경로를 찾는 것입니다. 승리 지점이 1개가 아니라, 상대편의 출발선에 있는 모든 칸이라는 점에 주의하세요.

If the given piece cannot be moved to the victory point, return an empty list [].

만약, 주어진 말을 승리지점까지 옮길 수 없다면, 빈 리스트 []를 반환하세요.

- Type: Individual Assignment
  형태: 개인과제

- Disqualification: You will be disqualified if any of the following conditions are violated:

  실격: 다음 조건 중 하나라도 **위반** 하였을 때

  1. You must implement one of the Uninformed search or Heuristic search algorithms exactly as described in the course materials, without any modifications. 
  
     수업 자료에 적힌 Uninformed search 또는 Heuristic search 중 하나를 변형 없이 구현해야 한다.
  2. The additional memory usage of your algorithm should not exceed 100MB.
  
     알고리즘의 추가 메모리 사용량이 100MB 이내여야 한다.
  
- Victory / 승리
  1. Basic Level: Find the shortest path that "runs without error" and is the same length or shorter than the path found by the teaching assistant’s Basic-level model.
  
    기초 단계: "오류 없이 실행가능한" 최단 경로를, 조교의 기초단계 모델과 동일하거나 더 짧은 길이로 찾으세요.

  2. Intermediate Level: Achieve the Basic level and manage memory usage to be within 1MB.
  
    중급 단계: 기초 단계를 달성하고, 메모리 사용량을 1MB 안쪽으로 관리하세요.

  3. Advanced Level: Achieve the Intermediate level and finish the search in the same or less time than the teaching assistant’s Advanced-level model.
  
    고급 단계: 중급 단계를 달성하고, 조교 고급단계 모델의 탐색시간과 동일하거나 더 짧은 길이로 탐색을 종료하세요.

  4. Challenge Level: Achieve the Advanced level and finish the search with memory usage less than or equal to the teaching assistant’s Advanced-level model.
  
    도전 단계: 고급 단계를 달성하고, 조교 고급단계 모델의 메모리 사용량 이하로 탐색을 종료하세요.

**Note**: The teaching assistant’s code is not publicly available, but by running your code alongside others' code in the "agents" folder, you can compare it directly with theirs. This will make it easier to implement the Advanced level. Additionally, you can always submit your code to the evaluation system, which will compare your code’s search actions with that of the teaching assistant and provide the results within 6 hours.

**참고**: 조교 코드는 공개되지 않지만, agents 폴더에 다른 사람의 코드를 받아 함께 실행하면 다른 사람의 코드와 동시에 비교할 수 있으니 고급단계를 구현하는 데 큰 무리가 없을 것입니다. 또한, 언제든 평가시스템에 여러분의 코드를 제출하면 평가 시스템이 여러분의 코드와 조교의 탐색 횟수를 비교하여 결과를 6시간 내로 알려줄겁니다.


#### Part 2, Local search

The goal of this problem is to find the most appropriate barrier positions that interfere with your opponent's path, given that your path is already determined. While searching, you will know the opponent's position, but you should not search for the opponent's shortest path. Additionally, your search must start from one of the adjacent edges to your current position, and you can only move to one of the 8 adjacent areas.

이 문제의 목표는 여러분의 경로가 결정된 상태에서, 상대방의 진로를 방해하는 가장 적절한 장벽 위치 조합을 찾는 것입니다. 탐색 시 상대방의 위치는 알고있겠지만, 여러분은 상대방의 최단경로를 탐색해서는 안됩니다. 또한, 여러분의 탐색은 여러분의 현 위치에 인접한 모서리에서 시작해야 하고, 인접한 8개 영역으로만 움직일 수 있습니다.

To obtain the shortest distance of your opponent's path, you should use the provided function `board.distance_to_goal()` (You cannot use your own function, to make a fair comparison of time/memory)

상대방의 최단경로 길이를 알고 싶다면, `board.distance_to_goal()` 함수를 사용하세요. (시간/메모리 계측의 공정성을 위해서, 각자 만든 함수로 측정할 수 없습니다.)

- Format: Individual task

  형태: 개인과제
  
  - Disqualification: You will be disqualified if any of the following conditions are **violated**:

    실격: 다음 조건 중 하나라도 **위반**하였을 때
  
    1. You must use the Local Search algorithm as described in the course materials, with or without modification, and heuristic/uninformed search cannot be combined.

       수업 자료에 적힌 Local search 알고리즘을 그대로 또는 변형하여 사용해야 하고, heuristic/uninformed search는 결합할 수 없습니다.
     
    2. You cannot search for the path that the opponent will actually take. To get the minimum distance of the opponent, use `board.distance_to_goal()`

       상대방이 실제로 이동할 경로 탐색은 할 수 없습니다. 상대방의 최단 경로 길이를 알고 싶다면, `board.distance_to_goal()`을 사용하세요.
     
    3. You should find a complete configuration where all of your fences installed (Not three fences anymore). For example, if you're starting with 7 fences left, you should find the position of all seven fences.
    
       탐색 결과로 모든 장벽이 설치된 완전한 조합을 찾아야 합니다 (이전처럼 세 장벽이 아닙니다). 예를 들어, 여러분이 7개의 장벽이 남아있다면, 그 7개 장벽을 모두 설치해야 합니다. 
 
    4. During the local search process, the number of fences should not be changed. You can only change the position of already placed fences, after the initial state.
  
       국소 탐색 과정에서, 장벽의 개수는 변할 수 없습니다. 초기 상태 이후에 여러분은 장벽의 위치를 바꿀수만 있습니다.
     
    5. During the search procedure, maximum three fences can be replaced when moving to other states. 
       So, whenever you change the state using `board.simulate_action()` or `board.set_state()` after initialization, 
       the system automatically checks whether the number of removed fences are at most five between the previous state and the new one. 

       탐색 과정 동안, 다른 상태로 변경하는 작업은 최대 5개의 장벽까지만 교체를 허용합니다.
       그러니까, 판 초기화 이후에 여러분이 `board.simulate_action()` 또는 `board.set_state()` 함수를 부를 때마다, 함수는 이전 상태와 현재 상태 사이에서 제거된 펜스가 최대 3개인지를 확인합니다.

    6. You cannot use parallel execution or parallel search, such as multi-threading or multi-processing.

       multi-threading이나 multi-processing 등 병렬 실행이나 병렬 탐색은 할 수 없습니다.
     
    7. The algorithm execution time must not exceed 5 minutes, and the memory usage must not exceed 5MB.

       알고리즘 구동 시간은 5분을 초과할 수 없고, 메모리 사용량은 5MB를 초과할 수 없습니다.
     
- Victory / 승리
  1. Basic Level: Find the optimal barrier positions* that are equal to or better than the TA model's within 5 minutes while satisfying the restricted conditions without errors.

     기초 단계: 제한된 조건을 오류 없이 만족하면서, 5분 이내에 조교 모델과 대등하거나 더 적합한 최적의 장벽 위치*를 찾으면 됩니다.
     
  2. Intermediate Level: Achieve the basic level and complete the search within 10 seconds.

     중급 단계: 기초 단계를 달성하고, 탐색을 60초 이내로 완료하세요.
     
  3. Advanced Level: Achieve the intermediate level and complete the search with memory usage within 1MB.

     고급 단계: 중급 단계를 달성하고, 메모리 사용량을 1MB 이내로 완료하세요.
     
  4. Challenge Level: Achieve the advanced level and find the optimal barrier position with fewer search steps than the TA model.

     도전 단계: 고급 단계를 달성하고, 조교 모델보다 적은 탐색 횟수로 최적 장벽 위치를 찾으세요.

**Note**: The assistant code is again disclosed. The barrier positions found by the assistant model are evaluated based on how much the opponent's shortest path increases, and the position that increases it the most is considered the most suitable barrier position. However, if the game rules are violated, you will be disqualified, and the opponent’s distance will be calculated as "0" instead of infinity. Similarly, when submitting your code to the evaluation system, you will receive the result within 6 hours.

**참고**: 조교 코드는 역시 공개되지 않습니다. 조교 모델이 찾은 장벽 위치와 대등하거나 더 적합한 위치는 상대방의 최단경로가 얼마나 증가하는지를 바탕으로 계산하며, 더 많이 증가한 경우에 더 적합한 장벽 위치로 판단합니다. 단, 게임 규칙에 위반되는 경우는 실격패를 당하게 되므로, 상대방의 거리는 무한대가 아니라 "0"으로 계산됩니다. 마찬가지로 평가 시스템에 코드를 제출하면 결과를 6시간 내로 알 수 있습니다.

**Note**: The program will check the existence of new fences using its center information. Thus, rotating a fence is not treated as a removal of a fence.

**참고**: 프로그램은 새로 세워진 장벽이 사라졌는지를 장벽의 중심 위치로만 판단합니다. 그러니까, 장벽을 회전하는 것은 장벽을 삭제하는 것으로 취급되지 않습니다.

#### Part 3. Belief-state search (disclosed)

#### Part 4. Adversarial Search (disclosed)

#### Environment (환경)

도전과제의 환경 구성은 기본적인 쿼리도 판 구성을 따릅니다.

In terms of seven characteristics, the game can be classified as:

환경을 기술하는 7개의 특징을 기준으로, 게임은 다음과 같이 분류됩니다:

- Fully observable (전체관측가능)

  You know everything required to decide your action.

  여러분은 이미 필요한 모든 내용을 알고 있습니다.

- Competitive Multi-agent (경쟁적 다중 에이전트)

  The other agents will do greedy actions to win the game.

  다른 에이전트들은 게임을 이기기 위하여 탐욕적 행동들을 수행합니다.

- Deterministic (결정론적)

  There's no unexpected chances of change on the board when executing the sequence of actions.

  행동을 순서대로 수행할 때, 예상치 못한 변수가 작용하지 않습니다.

- Sequential actions (순차적 행동)

  You should handle the sequence of your actions to play the game.

  게임 플레이를 위해서 필요한 여러분의 행동의 순서를 고민해야 합니다.

- Semi-dynamic performance (준역동적 지표)

  Some winning conditions are related to dynamic things, such as memory usage or time.

  승리조건의 일부 요소는 메모리나 시간의 영향을 받습니다.

- Discrete action, perception, state and time (이산적인 행동, 지각, 상태 및 시간개념)

  All of your actions, perceptions, states and time will be discrete, although you can query about your current memory usage in the computing procedure.

  여러분의 모든 행동, 지각, 상태 및 시간 흐름은 모두 이산적입니다. 여러분이 계산 도중 메모리 사용량을 시스템에 물어볼 수 있다고 하더라도 말입니다.

- Known rules (규칙 알려짐)

  All rules basically follows the original Quoridor game.

  모든 규칙은 기본적으로 원래의 쿼리도 게임을 따릅니다.

#### Actions

You can take one of the following actions.

다음 행동 중의 하나를 할 수 있습니다.

- **MOVE(direction)**: Move your piece to one of the four directions.

  상하좌우 방향 중 하나로 말 옮기기

- **BLOCK(position)**: Place a block at a position.

  특정한 모서리에 장벽 세워서 막기

  Here, the list of applicable edges will be given by the board.

  도로 짓기가 가능한 모서리의 목록은 board가 제공합니다.

#### Sensors

You can perceive the game state, during the search, as follows:

- The board (게임 판)
  - Coordinates of pieces and blocks

    모든 말과 장벽의 위치

  - You can ask the board to the list of applicable actions for.

    가능한 행동에 대해서 게임판 객체에 물어볼 수 있습니다.

- The number of total blocks remained (사용가능한, 남은 장벽 수)

## Structure of evaluation system

평가 시스템의 구조

The evaluation code has the following structure.

평가용 코드는 다음과 같은 구조를 가지고 있습니다.

```text
/                   ... The root of this project
/README.md          ... This README file
/evaluate.py        ... The entrance file to run the evaluation code
/board.py           ... The file that specifies programming interface with the board
/actions.py         ... The file that specifies actions to be called
/util.py            ... The file that contains several utilities for board and action definitions.
/agents             ... Directory that contains multiple agents to be tested.
/agents/__init__.py ... Helper code for loading agents to be evaluated
/agents/load.py     ... Helper code for loading agents to be evaluated
/agents/example.py  ... An example agent, provided by TA
/agents/_skeleton.py... A skeleton code for your agent. (You should change the name of file to run your code)
/evaluator/__init__.py. Code which executes evaluation of your agent.
/evaluator/part1.py ... Code for evaluating PART I.
/evaluator/part2.py ... Code for evaluating PART II.
/evaluator/part3.py ... Code for evaluating PART III.
/evaluator/part4.py ... Code for evaluating PART IV.
/evaluator/util.py  ... Helper code for evaluation.
```

All the codes have documentation that specifies what's happening on that code (only in English).

모든 코드는 어떤 동작을 하는 코드인지에 대한 설명이 달려있습니다 (단, 영어로만).

To deeply understand the `board.py` and `actions.py`, you may need some knowlege about [`pyquoridor` library](https://github.com/playquoridor/python-quoridor).

`board.py`와 `actions.py`를 깊게 이해하고 싶다면, [`pyquoridor` library](https://github.com/playquoridor/python-quoridor) 라이브러리에 대한 지식이 필요할 수 있습니다.

### What should I submit?

You should submit an agent python file, which has a similar structure to `/agents/default.py`.
That file should contain a class name `Agent` and that `Agent` class should have a method named `heuristic_search(board)`, `local_search(board, time_limit)`, `belief_state_search(board, time_limit)`, and `adversarial_search(board, time_limit)`.
Please use `/agents/_skeleton.py` as a skeleton code for your submission.

`/agents/default.py`와 비슷하게 생긴 에이전트 코드를 담은 파이썬 파일을 제출해야 합니다.
해당 코드는 `Agent`라는 클래스가 있어야 하고, `Agent` 클래스는 `heuristic_search(board)`, `local_search(board, time_limit)`, `belief_state_search(board, time_limit)` 및 `adversarial_search(board, time_limit)` 메서드를 가지고 있어야 합니다.
편의를 위해서 `/agents/_skeleton.py`를 골격 코드로 사용하여 제출하세요.

Also, you cannot use the followings to reduce your search time:

그리고 시간을 줄이기 위해서 다음에 나열하는 것을 사용하는 행위는 제한됩니다.

- multithreading / 멀티스레딩
- multiprocessing / 멀티프로세싱
- using other libraries other than basic python libraries. / 기본 파이썬 라이브러리 이외에 다른 라이브러리를 사용하는 행위

The TA will check whether you use those things or not. If so, then your evaluation result will be marked as zero.

조교가 여러분이 해당 사항을 사용하였는지 아닌지 확인하게 됩니다. 만약 그렇다면, 해당 평가 점수는 0점으로 처리됩니다.

## RUN

실행

To run the evaluation code, do the following:

1. (Only at the first run) Install the required libraries, by run the following code on your terminal or powershell, etc:

   (최초 실행인 경우만) 다음 코드를 터미널이나 파워쉘 등에서 실행하여, 필요한 라이브러리를 설치하세요.

    ```bash
    pip install -r requirements.txt
    ```

2. Place your code under `/agents` directory.

    여러분의 코드를 `/agents` 디렉터리 밑에 위치시키세요.

3. Execute the evaluation code, by run the following code on a terminal/powershell:

    다음 코드를 실행하여 평가 코드를 실행하세요.

    ```bash 
    python evaluate.py -p [PART]
    ```
   
    Here, `[PART]` indicates the part number. For example, if you want to execute evaluation for PART III, type:

    여기서, `[PART]`는 문제 번호입니다. 예를 들어, Part 3을 평가하고 싶다면, 아래와 같이 실행하세요:

    ```bash 
    python evaluate.py -p 3
    ```

    If you want to print out all computational procedure, then put `--debug` at the end of python call, as follows:

    만약, 모든 계산 과정을 출력해서 보고 싶다면, `--debug`을 파이썬 호출 부분 뒤에 붙여주세요.

    ```bash 
    python evaluate.py -p 3 --debug
    ```

4. See what's happening.

    어떤 일이 일어나는지를 관찰하세요.

Note: All the codes are tested both on (1) Windows 11 (23H2) with Python 3.9.13 and (2) Ubuntu 22.04 with Python 3.10. Sorry for Mac users, because you may have some unexpected errors.

모든 코드는 윈도우 11 (23H2)와 파이썬 3.9.13 환경과, 우분투 22.04와 파이썬 3.10 환경에서 테스트되었습니다. 예측불가능한 오류가 발생할 수도 있어, 미리 맥 사용자에게 미안하다는 말을 전합니다.
