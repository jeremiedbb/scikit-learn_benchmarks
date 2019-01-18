from sklearn.metrics.pairwise import pairwise_distances

from .common import Benchmark
from .datasets import _random_dataset


class PairwiseDistances_bench(Benchmark):
    """
    Benchmarks for pairwise distances.
    """

    param_names = ['representation', 'metric', 'n_jobs']
    params = (['dense', 'sparse'],
              ['cosine', 'euclidean', 'manhattan', 'correlation'],
              Benchmark.n_jobs_vals)

    def setup(self, *params):
        representation, metric, n_jobs = params

        if representation is 'sparse' and metric is 'correlation':
            raise NotImplementedError

        if Benchmark.data_size == 'large':
            if metric in ('manhattan', 'correlation'):
                n_samples = 8000
            else:
                n_samples = 32000
        else:
            if metric in ('manhattan', 'correlation'):
                n_samples = 4000
            else:
                n_samples = 16000

        self.X = _random_dataset(n_samples=n_samples,
                                 representation=representation)

        self.pdist_params = {'metric': metric,
                             'n_jobs': n_jobs}

    def time_pairwise_distances(self, *args):
        pairwise_distances(self.X, **self.pdist_params)

    def peakmem_pairwise_distances(self, *args):
        pairwise_distances(self.X, **self.pdist_params)
