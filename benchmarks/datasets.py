import numpy as np
import scipy.sparse as sp
from joblib import Memory

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import (load_sample_image, fetch_20newsgroups,
                              fetch_openml, load_digits, make_regression,
                              make_classification, fetch_olivetti_faces)
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# memory location for caching datasets
M = Memory(location='/tmp/joblib')


@M.cache
def _china_dataset(n_samples=None, dtype=np.float32):
    img = load_sample_image('china.jpg')
    X = np.array(img, dtype=dtype) / 255
    X = X.reshape((-1, 3))[:n_samples]
    return X


@M.cache
def _20newsgroups_highdim_dataset(n_samples=None, ngrams=(1, 1),
                                  dtype=np.float32):
    newsgroups = fetch_20newsgroups()
    vectorizer = TfidfVectorizer(ngram_range=ngrams)
    X = vectorizer.fit_transform(newsgroups.data[:n_samples])
    X = X.astype(dtype, copy=False)
    y = newsgroups.target[:n_samples]
    return X, y


@M.cache
def _20newsgroups_lowdim_dataset(n_components=100, ngrams=(1, 1),
                                 dtype=np.float32):
    newsgroups = fetch_20newsgroups()
    vectorizer = TfidfVectorizer(ngram_range=ngrams)
    X = vectorizer.fit_transform(newsgroups.data)
    X = X.astype(dtype, copy=False)
    svd = TruncatedSVD(n_components=n_components)
    X = svd.fit_transform(X)
    y = newsgroups.target
    return X, y


@M.cache
def _mnist_dataset(dtype=np.float32):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    return X, y


@M.cache
def _digits_dataset(n_samples=None, dtype=np.float32):
    X, y = load_digits(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    return X[:n_samples], y[:n_samples]


@M.cache
def _synth_regression_dataset(n_samples=1000, n_features=10000,
                              dtype=np.float32):
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=n_features // 10, noise=0.1,
                           random_state=0)
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)
    return X, y


@M.cache
def _synth_classification_dataset(n_samples=1000, n_features=10000,
                                  n_classes=2, dtype=np.float32):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, random_state=42,
                               n_informative=n_features, n_redundant=0)
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)
    return X, y


@M.cache
def _olivetti_faces_dataset():
    """
    Using example from documentation
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
    """
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    faces = dataset.data
    n_samples, n_features = faces.shape
    faces_centered = faces - faces.mean(axis=0)
    # local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    data = faces_centered

    return data


@M.cache
def _random_dataset(n_samples=1000, n_features=1000,
                    representation='dense', dtype=np.float32):
    if representation == 'dense':
        X = np.random.random_sample((n_samples, n_features))
        X = X.astype(dtype, copy=False)
    else:
        X = sp.random(n_samples, n_features, format='csr', dtype=dtype)
    return X
