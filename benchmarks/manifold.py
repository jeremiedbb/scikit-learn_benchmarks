from sklearn.manifold import TSNE

from .common import Benchmark
from .datasets import _digits_dataset


class TSNE_(Benchmark):
    """
    Benchmarks for t-SNE.
    """
    # params = (method)
    param_names = []
    params = ()

    def setup(self):
        self.X, _ = _digits_dataset()

        self.tsne_params = {'random_state': 0}

    def time_fit(self, *args):
        tsne = TSNE(**self.tsne_params)
        tsne.fit(self.X)

    def peakmem_fit(self, *args):
        tsne = TSNE(**self.tsne_params)
        tsne.fit(self.X)

    def track_fit(self, *args):
        tsne = TSNE(**self.tsne_params)
        tsne.fit(self.X)
        return tsne.n_iter_
