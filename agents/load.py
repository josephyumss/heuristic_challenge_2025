from pathlib import Path


def get_all_agents():
    """
    Load all agents in the current directory

    :return: List of agent file names
    """

    agents = []   #Path(__file__).parent  # >>> "/home/user/project"   glob는 특정 패턴의 파일을 찾아줌 지금은 .py
    for code in Path(__file__).parent.glob('*.py'):  # Query for all python files in this directory
        if code.stem != 'load' and not code.stem.startswith('_'):  # If the file name doesn't match with special file names,
            agents.append(code.stem)  # Include them    load.py 와 앞이 _로 시작하는 파일은 불러오지 않음

    return sorted(agents)


# Export only getter function
__all__ = ['get_all_agents']  #다른 패키지에서 import할 때 get all agents 만 가져올 수 있음
