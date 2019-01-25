from sklearn.manifold import TSNE

from .common import Benchmark, Estimator_bench
from .datasets import _digits_dataset


class TSNE_bench(Benchmark, Estimator_bench):
    """
    Benchmarks for t-SNE.
    """

    param_names = ['method']
    params = (['exact', 'barnes_hut'])

    def setup(self, *params):
        method, = params

        n_samples = 500 if method == 'exact' else None

        self.X, _, _, _ = _digits_dataset(n_samples=n_samples)

        self.estimator = TSNE(random_state=0, method=method)

        self.estimator.fit(self.X)
