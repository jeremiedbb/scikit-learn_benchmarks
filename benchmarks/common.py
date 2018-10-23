import os
import timeit

"""
if environment variable $MULTICORE is set, benchmarks will be run for
n_jobs vals = (1, $MULTICORE). Otherwise for n_jobs vals = (1, -1)
MULTICORE can be a list of values as well.
Example: export MULTICORE=2,3,4,5
"""

N_JOBS_VALS = [1, ]

try:
    multicore = os.getenv("MULTICORE")
    results = map(int, multicore.split(','))
    N_JOBS_VALS.extend(results)
except OSError:
    N_JOBS_VALS = [1, -1]


class Benchmark:
    timer = timeit.default_timer  # wall time
    timeout = 500
    processes = 1
    number = 1
    repeat = 1

    param_names = ["n_jobs"]
    params = (N_JOBS_VALS,)
