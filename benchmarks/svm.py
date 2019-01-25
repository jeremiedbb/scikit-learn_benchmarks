import numpy as np
from sklearn.svm import SVC

from .common import Benchmark, Estimator_bench, Predictor_bench
from .datasets import _synth_classification_dataset


def optimal_cache_size(n_features, dtype=np.float32):
    byte_size = np.empty(0, dtype=dtype).itemsize
    optimal_cache_size_bytes = n_features * n_features * byte_size
    eight_gb = 8e9
    cache_size_bytes = min(optimal_cache_size_bytes, eight_gb)
    return cache_size_bytes


class SVC_bench(Benchmark, Estimator_bench, Predictor_bench):
    """Benchmarks for SVC."""

    param_names = ['kernel']
    params = (['linear', 'poly', 'rbf', 'sigmoid'],)

    def setup(self, *params):
        kernel, = params

        self.X, _, self.y, _ = _synth_classification_dataset()

        self.estimator = SVC(C=0.01,
                             cache_size=optimal_cache_size(self.X.shape[1]),
                             max_iter=100,
                             tol=1e-16,
                             kernel=kernel,
                             random_state=0,
                             shrinking=True)

        self.estimator.fit(self.X, self.y)
