from sklearn.decomposition import PCA, DictionaryLearning

from .common import Benchmark
from .datasets import _mnist_dataset


class PCA_(Benchmark):
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


class DictionaryLearning_(Benchmark):
    """
    Benchmarks for DictionnaryLearning.
    """
    pass
