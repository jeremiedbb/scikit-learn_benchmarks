import os


# if environment variable $MULTICORE is set, benchmarks will be run for
# n_jobs vals = (1, $MULTICORE). Otherwise for n_jobs vals = (1, -1)
N_JOBS_VALS = [1, int(os.getenv('MULTICORE', -1))]


class Benchmark():
    param_names = ['n_jobs']
    params = (N_JOBS_VALS,)
