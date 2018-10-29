import numpy as np
import scipy.sparse as sp

from joblib import Memory

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import (load_sample_image, fetch_openml,
                              fetch_20newsgroups, load_digits,
                              make_regression, make_classification)
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# memory location for caching datasets
M = Memory(location="/tmp/joblib")


@M.cache
def _china_dataset(dtype=np.float32):
    img = load_sample_image("china.jpg")
    X = np.array(img, dtype=dtype) / 255
    X = X.reshape((-1, 3))
    return X


@M.cache
def _20newsgroups_highdim_dataset(dtype=np.float32):
    newsgroups = fetch_20newsgroups()
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.9)
    X = vectorizer.fit_transform(newsgroups.data)
    X = X.astype(dtype, copy=False)
    y = newsgroups.target
    return X, y


@M.cache
def _20newsgroups_lowdim_dataset(n_components=100, dtype=np.float32):
    newsgroups = fetch_20newsgroups()
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.9)
    X = vectorizer.fit_transform(newsgroups.data)
    X = X.astype(dtype, copy=False)
    svd = TruncatedSVD(n_components=n_components)
    X = svd.fit_transform(X)
    y = newsgroups.target
    return X, y


@M.cache
def _mnist_dataset(dtype=np.float32):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    return X, y


@M.cache
def _digits_dataset(dtype=np.float32):
    X, y = load_digits(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    return X, y


@M.cache
def _synth_regression_dataset(n_samples=1000, n_features=10000,
                              representation="dense", dtype=np.float32):
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=n_features//10, noise=0.1)
    X = X.astype(dtype, copy=False)

    if representation is 'sparse':
        X[X < 2] = 0
        X = sp.csr_matrix(X)

    return X, y


@M.cache
def _synth_classification_dataset(n_samples=1000, n_features=10000,
                                  representation='dense', n_classes=2,
                                  dtype=np.float32):

    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, random_state=42,
                               n_informative=n_features, n_redundant=0)

    X = X.astype(dtype, copy=False)

    if representation is 'sparse':
        X[X < 2] = 0
        X = sp.csr_matrix(X)

    X = MaxAbsScaler().fit_transform(X)
    return X, y


def _random_dataset(n_samples=1000, n_features=1000,
                    representation='dense', dtype=np.float32):
    if representation is 'dense':
        X = np.random.random_sample((n_samples, n_features))
        X = X.astype(dtype, copy=False)
    else:
        X = sp.random(n_samples, n_features, format='csr', dtype=dtype)
    return X
