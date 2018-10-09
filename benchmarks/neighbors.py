from sklearn.neighbors import KNeighborsClassifier

from .common import Benchmark
from .datasets import _20newsgroups_lowdim_dataset


class KNeighborsClassifier_bench(Benchmark):
    """
    Benchmarks for KNeighborsClassifier.
    """
    # params = (algorithm)
    param_names = ['params'] + Benchmark.param_names
    params = ([('brute',),
               ('kd_tree',),
               ('ball_tree',)],) + Benchmark.params

    def setup(self, params, *common):
        algo = params[0]

        n_jobs = common[0]

        self.X, self.y = _20newsgroups_lowdim_dataset()

        self.knn_params = {'algorithm': algo,
                           'n_jobs': n_jobs}

    def time_fit_predict(self, *args):
        knn = KNeighborsClassifier(**self.knn_params)
        knn.fit(self.X, self.y)
        knn.predict(self.X)

    def peakmem_fit_predict(self, *args):
        knn = KNeighborsClassifier(**self.knn_params)
        knn.fit(self.X, self.y)
        knn.predict(self.X)
