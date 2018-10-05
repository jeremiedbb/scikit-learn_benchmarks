import numpy as np

from joblib import Memory

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import (load_sample_image, fetch_20newsgroups_vectorized,
                              fetch_openml)


# memory location for caching datasets
M = Memory(location='/tmp/joblib')

load_sample_image = M.cache(load_sample_image)
fetch_20newsgroups_vectorized = M.cache(fetch_20newsgroups_vectorized)
fetch_openml = M.cache(fetch_openml)


def _china_dataset(dtype=np.float32):
    img = load_sample_image("china.jpg")
    X = np.array(img, dtype=dtype) / 255
    X = X.reshape((-1, 3))
    return X


def _20newsgroups_highdim_dataset(dtype=np.float32):
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    X = X.astype(dtype, copy=False)
    return X, y


def _20newsgroups_lowdim_dataset(dtype=np.float32):
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    X = X.astype(dtype, copy=False)
    svd = TruncatedSVD(n_components=10)
    X = svd.fit_transform(X)
    return X, y


def _mnist_dataset(dtype=np.float32):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.astype(dtype, copy=False)
    return X, y
