from sklearn.metrics.pairwise import pairwise_distances

from .common import Benchmark
from .datasets import _random_dataset


class PairwiseDistances_bench(Benchmark):
    """
    Benchmarks for pairwise distances.
    """
    # params = (representation, metric)
    param_names = ['params'] + Benchmark.param_names
    params = ([('dense', 'cosine'),
               ('sparse', 'cosine'),
               ('dense', 'euclidean'),
               ('sparse', 'euclidean'),
               ('dense', 'manhattan'),
               ('sparse', 'manhattan'),
               ('dense', 'correlation')],) + Benchmark.params

    def setup(self, params, *common):
        representation = params[0]
        metric = params[1]

        n_jobs = common[0]

        if metric in ('manhattan', 'correlation'):
            n_samples = 2000
        else:
            n_samples = 10000

        self.X = _random_dataset(n_samples=n_samples,
                                 representation=representation)

        self.pdist_params = {'metric': metric,
                             'n_jobs': n_jobs}

    def time_pairwise_distances(self, *args):
        pairwise_distances(self.X, **self.pdist_params)

    def peakmem_pairwise_distances(self, *args):
        pairwise_distances(self.X, **self.pdist_params)
