from sklearn.decomposition import (PCA, DictionaryLearning,
                                   MiniBatchDictionaryLearning)

from .common import Benchmark, Estimator_bench, Transformer_bench
from .datasets import _olivetti_faces_dataset, _mnist_dataset


class PCA_bench(Benchmark, Estimator_bench, Transformer_bench):
    """
    Benchmarks for PCA.
    """

    param_names = ['svd_solver']
    params = (['full', 'arpack', 'randomized'],)

    def setup(self, *params):
        svd_solver, = params

        self.X, _, _, _ = _mnist_dataset()

        self.estimator = PCA(n_components=8,
                             svd_solver=svd_solver,
                             random_state=0)

        self.estimator.fit(self.X)


class DictionaryLearning_bench(Benchmark, Estimator_bench, Transformer_bench):
    """
    Benchmarks for DictionaryLearning.
    """

    param_names = ['fit_algorithm', 'n_jobs']
    params = (['lars', 'cd'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        fit_algorithm, n_jobs = params

        self.X, _ = _olivetti_faces_dataset()

        self.estimator = DictionaryLearning(n_components=15,
                                            fit_algorithm=fit_algorithm,
                                            alpha=0.1,
                                            max_iter=20,
                                            tol=1e-16,
                                            random_state=0,
                                            n_jobs=n_jobs)

        self.estimator.fit(self.X)


class MiniBatchDictionaryLearning_bench(Benchmark, Estimator_bench,
                                        Transformer_bench):
    """
    Benchmarks for MiniBatchDictionaryLearning
    """

    param_names = ['fit_algorithm', 'n_jobs']
    params = (['lars', 'cd'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        fit_algorithm, n_jobs = params

        self.X, _ = _olivetti_faces_dataset()

        self.estimator = MiniBatchDictionaryLearning(
                            n_components=15,
                            fit_algorithm=fit_algorithm,
                            alpha=0.1,
                            batch_size=3,
                            random_state=0,
                            n_jobs=n_jobs)

        self.estimator.fit(self.X)
