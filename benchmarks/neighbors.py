from sklearn.neighbors import KNeighborsClassifier

from .common import Benchmark
from .datasets import _20newsgroups_lowdim_dataset


class KNeighborsClassifier_bench(Benchmark):
    """
    Benchmarks for KNeighborsClassifier.
    """

    param_names = ['algorithm', 'dimension', 'n_jobs']
    params = (['brute', 'kd_tree', 'ball_tree'],
              ['high', 'low'],
              Benchmark.n_jobs_vals)

    def setup(self, *params):
        algorithm, dimension, n_jobs = params

        if dimension is 'high':
            self.X, self.y = _20newsgroups_lowdim_dataset(n_components=50)
        else:
            self.X, self.y = _20newsgroups_lowdim_dataset(n_components=10)

        self.knn_params = {'algorithm': algorithm,
                           'n_jobs': n_jobs}

    def time_fit_predict(self, *args):
        knn = KNeighborsClassifier(**self.knn_params)
        knn.fit(self.X, self.y)
        knn.predict(self.X)

    def peakmem_fit_predict(self, *args):
        knn = KNeighborsClassifier(**self.knn_params)
        knn.fit(self.X, self.y)
        knn.predict(self.X)
