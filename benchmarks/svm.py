import numpy as np
from sklearn import svm

from .common import Benchmark
from .datasets import _synth_classification_dataset


def get_optimal_cache_size(n_features, dtype=np.float32):
    byte_size = np.empty(0, dtype=dtype).itemsize
    optimal_cache_size_bytes = n_features * n_features * byte_size
    eight_gb = 8e9
    cache_size_bytes = min(optimal_cache_size_bytes, eight_gb)
    return cache_size_bytes


class SVCS_bench(Benchmark):

    # params = (Kernel)
    param_names = ['params']
    params = ([('linear',),
               ('poly',),
               ('rbf',),
               ('sigmoid',)],)

    def setup(self, params):
        self.X, self.y = _synth_classification_dataset()
        self.cache_size = get_optimal_cache_size(self.X.shape[1])

        self.svc_params = {
            "C": 0.01,
            "cache_size": self.cache_size,
            "max_iter": 100,
            "tol": 1e-16,
            "kernel": params[0],
            "random_state": 42,
            "shrinking": True,
        }

    def time_fit(self, *args):
        svc = svm.SVC(**self.svc_params)
        svc.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        svc = svm.SVC(**self.svc_params)
        svc.fit(self.X, self.y)
