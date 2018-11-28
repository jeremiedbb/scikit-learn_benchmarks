import os
import timeit
"""
if environment variable $MULTICORE is set, benchmarks will be run for
n_jobs vals = (1, $MULTICORE). Otherwise for n_jobs vals = (1, -1)
MULTICORE can be a list of values as well.
Example: export MULTICORE=2,3,4,5
"""

N_JOBS_VALS = [1, -1]

if os.getenv('MULTICORE'):
    multicore = os.getenv('MULTICORE')
    N_JOBS_VALS = list(map(int, multicore.split(',')))


class Benchmark:
    timer = timeit.default_timer  # wall time
    warmup_time = 1
    timeout = 500
    processes = 1
    sample_time = 0.5
    repeat = (3, 100, 120.0)

    param_names = ['n_jobs']
    params = (N_JOBS_VALS,)
