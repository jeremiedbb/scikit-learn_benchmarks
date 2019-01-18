from sklearn.decomposition import (PCA, DictionaryLearning,
                                   MiniBatchDictionaryLearning)

from .common import Benchmark
from .datasets import _olivetti_faces_dataset, _mnist_dataset


class PCA_bench(Benchmark):
    """
    Benchmarks for PCA.
    """

    param_names = ['svd_solver']
    params = (['full', 'arpack', 'randomized'],)

    def setup(self, *params):
        svd_solver, = params

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

    param_names = ['fit_algorithm', 'n_jobs']
    params = (['lars', 'cd'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        fit_algorithm, n_jobs = params

        self.data = _olivetti_faces_dataset()

        self.dl_params = {'n_components': 15,
                          'fit_algorithm': fit_algorithm,
                          'alpha': 0.1,
                          'max_iter': 20,
                          'tol': 1e-16,
                          'random_state': 42,
                          'n_jobs': n_jobs}

    def time_fit(self, *args):
        dict_learning = DictionaryLearning(**self.dl_params)
        dict_learning.fit(self.data)

    def peakmem_fit(self, *args):
        dict_learning = DictionaryLearning(**self.dl_params)
        dict_learning.fit(self.data)


class MiniBatchDictionaryLearning_bench(Benchmark):
    """
    Benchmarks for MiniBatchDictionaryLearning
    """

    param_names = ['fit_algorithm', 'n_jobs']
    params = (['lars', 'cd'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        fit_algorithm, n_jobs = params

        self.data = _olivetti_faces_dataset()

        self.dl_params = {'n_components': 15,
                          'fit_algorithm': fit_algorithm,
                          'alpha': 0.1,
                          'batch_size': 3,
                          'random_state': 42,
                          'n_jobs': n_jobs}

    def time_fit(self, *args):
        dict_learning = MiniBatchDictionaryLearning(**self.dl_params)
        dict_learning.fit(self.data)

    def peakmem_fit(self, *args):
        dict_learning = MiniBatchDictionaryLearning(**self.dl_params)
        dict_learning.fit(self.data)
