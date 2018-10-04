import numpy as np

from .common import Benchmark

from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _k_init
from sklearn.datasets import load_sample_image
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.utils.extmath import row_norms


def _dense_dataset():
    img = load_sample_image("china.jpg")
    X = np.array(img, dtype=np.float64) / 255
    X = X.reshape((-1, 3))
    n_clusters = 64
    return X, n_clusters


def _sparse_dataset():
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    n_clusters = len(set(y))
    return X, n_clusters


class Kmeans(Benchmark):
    """
    Benchmarks for KMeans.
    """
    # params = (representation, algorithm)
    param_names = ['params'] + Benchmark.param_names
    params = ([('dense', 'full'),
               ('dense', 'elkan'),
               ('sparse', 'full')],) + Benchmark.params

    def setup(self, params, *common):
        representation = params[0]
        algo = params[1]

        n_jobs = common[0]

        if representation is 'sparse':
            self.X, self.n_clusters = _sparse_dataset()
        else:
            self.X, self.n_clusters = _dense_dataset()

        self.x_squared_norms = row_norms(self.X, squared=True)
        self.rng = np.random.RandomState(0)

        self.kmeans_params = {'n_clusters': self.n_clusters,
                              'algorithm': algo,
                              'n_init': 1,
                              'n_jobs': n_jobs,
                              'random_state': self.rng}

    def time_iterations(self, *args):
        kmeans = KMeans(init='random', max_iter=10, tol=0,
                        **self.kmeans_params)
        kmeans.fit(self.X)

    def peakmem_iterations(self, *args):
        kmeans = KMeans(init='random', max_iter=10, tol=0,
                        **self.kmeans_params)
        kmeans.fit(self.X)

    def track_iterations(self, *args):
        kmeans = KMeans(init='random', max_iter=10, tol=0,
                        **self.kmeans_params)
        kmeans.fit(self.X)
        return kmeans.n_iter_

    def time_convergence(self, *args):
        kmeans = KMeans(**self.kmeans_params)
        kmeans.fit(self.X)

    def peakmem_convergence(self, *args):
        kmeans = KMeans(**self.kmeans_params)
        kmeans.fit(self.X)

    def track_convergence(self, *args):
        kmeans = KMeans(**self.kmeans_params)
        kmeans.fit(self.X)
        return kmeans.n_iter_

    def time_init(self, *args):
        _k_init(self.X, self.n_clusters, self.x_squared_norms,
                random_state=self.rng)

    def peakmem_init(self, *args):
        _k_init(self.X, self.n_clusters, self.x_squared_norms,
                random_state=self.rng)
