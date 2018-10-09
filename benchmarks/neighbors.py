from sklearn.neighbors import KNeighborsClassifier

from .common import Benchmark
from .datasets import _20newsgroups_lowdim_dataset


class KNeighborsClassifier_bench(Benchmark):
    """
    Benchmarks for KNeighborsClassifier.
    """
    # params = (algorithm, wideness)
    param_names = ['params'] + Benchmark.param_names
    params = ([('brute', 'high_dim'),
               ('brute', 'low_dim'),
               ('kd_tree', 'high_dim'),
               ('kd_tree', 'low_dim'),
               ('ball_tree', 'high_dim'),
               ('ball_tree', 'low_dim')],) + Benchmark.params

    def setup(self, params, *common):
        algo = params[0]
        wideness = params[1]

        n_jobs = common[0]

        if wideness is 'high_dim':
            self.X, self.y = _20newsgroups_lowdim_dataset(n_components=50)
        else:
            self.X, self.y = _20newsgroups_lowdim_dataset(n_components=10)

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
