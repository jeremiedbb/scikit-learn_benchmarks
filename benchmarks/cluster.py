import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms

from .common import (Benchmark, Estimator_bench,
                     Predictor_bench, Transformer_bench)
from .datasets import _china_dataset, _20newsgroups_highdim_dataset


class KMeans_bench(Benchmark, Estimator_bench,
                   Predictor_bench, Transformer_bench):
    """
    Benchmarks for KMeans.
    """

    param_names = ['representation', 'algorithm', 'n_jobs']
    params = (['dense', 'sparse'], ['full', 'elkan'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        representation, algorithm, n_jobs = params

        if representation == 'sparse' and algorithm == 'elkan':
            raise NotImplementedError

        if Benchmark.data_size == 'large':
            if representation == 'sparse':
                self.X, _, _, _ = _20newsgroups_highdim_dataset(ngrams=(1, 2))
                n_clusters = 20
            else:
                self.X, _ = _china_dataset()
                n_clusters = 256
        else:
            if representation == 'sparse':
                self.X, _, _, _ = _20newsgroups_highdim_dataset(n_samples=5000)
                n_clusters = 20
            else:
                self.X, _ = _china_dataset(n_samples=200000)
                n_clusters = 64

        self.estimator = KMeans(n_clusters=n_clusters,
                                algorithm=algorithm,
                                n_init=1,
                                init='random',
                                max_iter=50,
                                tol=1e-16,
                                n_jobs=n_jobs,
                                random_state=0)

        self.estimator.fit(self.X)


class KMeansPlusPlus_bench(Benchmark):
    """
    Benchmarks for k-means++ init.
    """

    def setup(self):
        self.X, _ = _china_dataset()
        self.n_clusters = 256
        self.x_squared_norms = row_norms(self.X, squared=True)

    def time_kmeansplusplus(self):
        _k_init(self.X, self.n_clusters, self.x_squared_norms,
                random_state=np.random.RandomState(0))

    def peakmem_kmeansplusplus(self):
        _k_init(self.X, self.n_clusters, self.x_squared_norms,
                random_state=np.random.RandomState(0))
