import os
from multiprocessing import cpu_count
import json
import timeit


def get_from_config():
    config_path = os.path.dirname(os.path.realpath(__file__))+"/config.json"

    with open(config_path, 'r') as config_file:
        config_file = "".join(line for line in config_file
                              if line and '//' not in line)
        config = json.loads(config_file)

        profile = config['profile']

        n_jobs_vals = config['n_jobs_vals']
        if not n_jobs_vals:
            n_jobs_vals = list(range(1, 1 + cpu_count()))

    return profile, n_jobs_vals


class Benchmark:

    timer = timeit.default_timer  # wall time
    warmup_time = 1
    processes = 1
    timeout = 500

    profile, n_jobs_vals = get_from_config()

    if profile == 'fast':
        repeat = 1
        number = 1
        data_size = 'small'
    elif profile == 'regular':
        repeat = (3, 100, 30.0)
        data_size = 'small'
    elif profile == 'large_scale':
        repeat = 3
        data_size = 'large'

    def __init__(self):
        self.X = None
        self.y = None


class Estimator_bench:
    def setup(self):
        raise NotImplementedError

    def time_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        self.estimator.fit(self.X, self.y)


class Predictor_bench:
    def setup(self):
        raise NotImplementedError

    def time_predict(self, *args):
        self.estimator.predict(self.X)

    def peakmem_predict(self, *args):
        self.estimator.predict(self.X)


class Transformer_bench:
    def setup(self):
        raise NotImplementedError

    def time_transform(self, *args):
        self.estimator.transform(self.X)

    def peakmem_transform(self, *args):
        self.estimator.transform(self.X)
