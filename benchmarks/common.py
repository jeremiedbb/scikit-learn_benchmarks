import os
from multiprocessing import cpu_count
import json
import timeit


CONFIG_PATH = os.path.dirname(os.path.realpath(__file__))+"/config.json"


class Benchmark:

    timer = timeit.default_timer  # wall time
    warmup_time = 1
    processes = 1
    timeout = 500

    # get common attributes from config file
    with open(CONFIG_PATH, 'r') as config_file:
        config_file = "".join(line for line in config_file
                              if line and '//' not in line)
        config = json.loads(config_file)

        profile = config['profile']

        n_jobs_vals = config['n_jobs_vals']
        if not n_jobs_vals:
            n_jobs_vals = list(range(1, 1 + cpu_count()))

    if profile == 'fast':
        repeat = 1
        number = 1
        data_size = 'small'
    elif profile == 'regular':
        repeat = (3, 100, 30.0)
        data_size = 'small'
    elif profile == 'large_scale':
        # repeat = (3, ?, ?)
        data_size = 'large'
