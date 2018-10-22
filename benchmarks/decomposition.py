from sklearn.decomposition import PCA, DictionaryLearning, MiniBatchDictionaryLearning

from .common import Benchmark
from .datasets import _mnist_dataset, _decomposition_dataset


class PCA_bench(Benchmark):
    """
    Benchmarks for PCA.
    """

    # params = (svd_solver)
    param_names = ["params"]
    params = ([("full",), ("arpack",), ("randomized",)],)

    def setup(self, params):
        svd_solver = params[0]

        self.X, _ = _mnist_dataset()

        self.pca_params = {
            "n_components": 8,
            "svd_solver": svd_solver,
            "random_state": 0,
        }

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

    def setup(self, params):
        self.data = _decomposition_dataset()

        self.estimator = DictionaryLearning(
            n_components=15,
            alpha=0.1,
            tol=1e-16,
            random_state=42,
            n_jobs=-1,
        )

    def time_fit(self, params):
        self.estimator.fit(self.data)


class MiniBatchDictionaryLearningSuite(Benchmark):
    def setup(self, params):

        self.data = _decomposition_dataset()
        self.estimator = MiniBatchDictionaryLearning(
            n_components=15,
            alpha=0.1,
            n_iter=50,
            batch_size=3,
            random_state=42,
            n_jobs=-1,
        )

    def time_fit(self, params):
        self.estimator.fit(self.data)
