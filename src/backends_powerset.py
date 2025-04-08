'''
This file contains the implementation of the function for the backend distribution.
The distribution must implement the following function:
    - backend_distribution(backends) -> (backend_distribution)
        backends: list of tuples (provider, backend) representing all the considered backends
        backend_distribution: list of tuples (provider, backend) where each tuple will be used in the experiments
'''

from itertools import chain, combinations

def backend_distribution(backends):
    return powerset(backends)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return  list(chain.from_iterable(combinations(s, r) for r in range(2,len(s)+1)))
