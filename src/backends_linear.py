'''
This file contains the implementation of the function for the backend distribution.
The distribution must implement the following function:
    - backend_distribution(backends) -> (backend_distribution)
        backends: list of tuples (provider, backend) representing all the considered backends
        backend_distribution: list of tuples (provider, backend) where each tuple will be used in the experiments
'''
import random

# Number of list equal to the number of backends-1 (from 2), each time sampled randomly
def backend_distribution(backends):
    num_backends = len(backends)
    backend_to_use = []
    for i in range(2, num_backends+1):
        backend_to_use.append(random.sample(backends, i))
    return backend_to_use


