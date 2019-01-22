from sklearn.manifold import TSNE

from .common import Benchmark
from .datasets import _digits_dataset


class TSNE_bench(Benchmark):
    """
    Benchmarks for t-SNE.
    """

    param_names = ['method']
    params = (['exact', 'barnes_hut'])

    def setup(self, *params):
        method, = params

        n_samples = 500 if method == 'exact' else None

        self.X, _ = _digits_dataset(n_samples=n_samples)

        self.tsne_params = {'random_state': 0,
                            'method': method}

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
