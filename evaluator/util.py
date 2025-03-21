from collections import namedtuple
from traceback import print_exc

import psutil

#: Size of MB in bytes
MEGABYTES = 1024 ** 2


# Named tuples for recording
Performance = namedtuple('Performance',
                         [
                             'failure', # Reason of failure.
                             'outcome', # Outcome of performance metric
                             'time', # Time for execution
                             'search', # Amount of search actions
                             'memory', # Memory consumption
                             'point'])  # Point earned (basic/intermediate/high)


def load_ta_agent():
    try:
        from importlib import import_module
        module = import_module('agents._ta')
        return module.Agent
    except:
        return None

def load_random_agent():
    try:
        from importlib import import_module
        module = import_module('agents._random')
        return module.Agent
    except:
        return None