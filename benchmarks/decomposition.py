from sklearn.decomposition import (PCA, DictionaryLearning,
                                   MiniBatchDictionaryLearning)

from .common import Benchmark
from .datasets import _decomposition_dataset, _mnist_dataset


class PCA_bench(Benchmark):
    """
    Benchmarks for PCA.
    """

    # params = (svd_solver)
    param_names = ['params']
    params = ([('full',),
               ('arpack',),
               ('randomized',)],)

    def setup(self, params):
        svd_solver = params[0]

        self.X, _ = _mnist_dataset()

        self.pca_params = {'n_components': 8,
                           'svd_solver': svd_solver,
                           'random_state': 0}

    def time_fit(self, *args):
        pca = PCA(**self.pca_params)
        pca.fit(self.X)

    def peakmem_fit(self, *args):
        pca = PCA(**self.pca_params)
        pca.fit(self.X)


class DictionaryLearning_bench(Benchmark):
    """
    Benchmarks for DictionaryLearning.
    """

    # params = (fit_algorithm)
    param_names = ['params'] + Benchmark.param_names
    params = ([('lars',),
               ('cd',)],) + Benchmark.params

    def setup(self, params, *common):
        self.data = _decomposition_dataset()
        fit_algorithm = params[0]
        n_jobs = common[0]
        self.dl_params = {'n_components': 15,
                          'fit_algorithm': fit_algorithm,
                          'alpha': 0.1,
                          'tol': 1e-16,
                          'random_state': 42,
                          'n_jobs': n_jobs}

    def time_fit(self, params):
        estimator = DictionaryLearning(**self.dl_params)
        estimator.fit(self.data)

    def peakmem_fit(self, params):
        estimator = DictionaryLearning(**self.dl_params)
        estimator.fit(self.data)


class MiniBatchDictionaryLearning_bench(Benchmark):
    """
    Benchmarks for MiniBatchDictionaryLearning
    """
    # params = (fit_algorithm)
    param_names = ['params'] + Benchmark.param_names
    params = ([('lars',),
               ('cd',)],) + Benchmark.params

    def setup(self, params, *common):
        self.data = _decomposition_dataset()
        fit_algorithm = params[0]
        n_jobs = common[0]
        self.dl_params = {'n_components': 15,
                          'fit_algorithm': fit_algorithm,
                          'alpha': 0.1,
                          'tol': 1e-16,
                          'random_state': 42,
                          'n_jobs': n_jobs}

    def time_fit(self, params):
        estimator = MiniBatchDictionaryLearning(**self.dl_params)
        estimator.fit(self.data)

    def peakmem_fit(self, params):
        estimator = MiniBatchDictionaryLearning(**self.dl_params)
        estimator.fit(self.data)
