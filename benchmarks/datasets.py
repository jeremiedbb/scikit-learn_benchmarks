import numpy as np
import scipy.sparse as sp

from joblib import Memory

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import (load_sample_image, fetch_20newsgroups_vectorized,
                              fetch_openml, load_digits, make_regression)
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


# memory location for caching datasets
M = Memory(location='/tmp/joblib')


@M.cache
def _china_dataset(dtype=np.float32):
    img = load_sample_image("china.jpg")
    X = np.array(img, dtype=dtype) / 255
    X = X.reshape((-1, 3))
    return X


@M.cache
def _20newsgroups_highdim_dataset(dtype=np.float32):
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    return X, y


@M.cache
def _20newsgroups_lowdim_dataset(dtype=np.float32):
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    svd = TruncatedSVD(n_components=10)
    X = svd.fit_transform(X)
    return X, y


@M.cache
def _mnist_dataset(dtype=np.float32):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)
    return X, y


@M.cache
def _digits_dataset(dtype=np.float32):
    X, y = load_digits(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)
    return X, y


@M.cache
def _synth_regression_lowdim_dataset(n_samples=1000000, n_features=100,
                                     dtype=np.float32):
    X, y = make_regression(n_samples=n_samples, n_features=n_features)
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)
    return X, y


@M.cache
def _synth_regression_highdim_dataset(n_samples=10000, n_features=100000,
                                      dtype=np.float32):
    X = sp.random(n_samples, n_features, density=0.001, format='csr',
                  dtype=dtype)
    y = np.random.rand(n_samples)
    return X, y
