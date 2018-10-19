import numpy as np
from sklearn import svm

from common import Benchmark
from datasets import _synth_classification_dataset


def getOptimalCacheSize(numFeatures):
    byte_size = np.empty(0, dtype=np.double).itemsize
    optimal_cache_size_bytes = numFeatures * numFeatures * byte_size
    eight_gb = byte_size * 1024 * 1024 * 1024
    cache_size_bytes = (
        eight_gb if optimal_cache_size_bytes > eight_gb else optimal_cache_size_bytes
    )
    return cache_size_bytes


class SVCSuite(Benchmark):
    def setup(self):
        self.X, self.y = _synth_classification_dataset()
        self.cache_size = getOptimalCacheSize(self.X.shape()[1])
        # We initialize classifier in `setup` to avoid its influence on timing
        self.clf = svm.SVC(
            C=0.01,
            cache_size=self.cache_size,
            max_iter=2000,
            tol=1e-16,
            kernel="linear",
            random_state=42,
            shrinking=True,
        )

    def time_fit(self):
        svc = svm.SVC(
            C=0.01,
            cache_size=self.cache_size,
            max_iter=2000,
            tol=1e-16,
            kernel="linear",
            random_state=42,
            shrinking=True,
        )

        svc.fit(self.X, self.y)

    def time_svm_fit_initialized(self):
        self.clf.fit(self.X, self.y)
