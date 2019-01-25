from sklearn.neighbors import KNeighborsClassifier

from .common import Benchmark, Estimator_bench, Predictor_bench
from .datasets import _20newsgroups_lowdim_dataset


class KNeighborsClassifier_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmarks for KNeighborsClassifier.
    """

    param_names = ['algorithm', 'dimension', 'n_jobs']
    params = (['brute', 'kd_tree', 'ball_tree'],
              ['low', 'high'],
              Benchmark.n_jobs_vals)

    def setup(self, *params):
        algorithm, dimension, n_jobs = params

        if Benchmark.data_size == 'large':
            nc = 40 if dimension == 'low' else 200
        else:
            nc = 10 if dimension == 'low' else 50

        self.X, _, self.y, _ = _20newsgroups_lowdim_dataset(n_components=nc)

        self.estimator = KNeighborsClassifier(algorithm=algorithm,
                                              n_jobs=n_jobs)

        self.estimator.fit(self.X, self.y)
