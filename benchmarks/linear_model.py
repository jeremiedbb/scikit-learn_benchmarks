from sklearn.linear_model import (LogisticRegression, Ridge, ElasticNet, Lasso,
                                  LinearRegression, SGDRegressor)

from .common import Benchmark
from .datasets import (_20newsgroups_highdim_dataset,
                       _20newsgroups_lowdim_dataset, _synth_regression_dataset)


class LogisticRegression_bench(Benchmark):
    """
    Benchmarks for LogisticRegression.
    """

    param_names = ['representation', 'solver', 'n_jobs']
    params = (['dense', 'sparse'], ['lbfgs', 'saga'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        representation, solver, n_jobs = params

        if Benchmark.data_size == 'large':
            if representation is 'sparse':
                self.X, self.y = _20newsgroups_highdim_dataset(n_samples=10000)
            else:
                self.X, self.y = _20newsgroups_lowdim_dataset(n_components=1e3)
        else:
            if representation is 'sparse':
                self.X, self.y = _20newsgroups_highdim_dataset(n_samples=2500)
            else:
                self.X, self.y = _20newsgroups_lowdim_dataset()

        if solver is 'lbfgs':
            self.lr_params = {'penalty': 'l2'}
        else:
            self.lr_params = {'penalty': 'l1'}

        self.lr_params.update({'solver': solver,
                               'multi_class': 'multinomial',
                               'tol': 0.01,
                               'n_jobs': n_jobs,
                               'random_state': 0})

    def time_fit(self, *args):
        lr = LogisticRegression(**self.lr_params)
        lr.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        lr = LogisticRegression(**self.lr_params)
        lr.fit(self.X, self.y)

    def track_fit(self, *args):
        lr = LogisticRegression(**self.lr_params)
        lr.fit(self.X, self.y)
        return int(lr.n_iter_[0])


class Ridge_bench(Benchmark):
    """
    Benchmarks for Ridge.
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        if representation is 'dense':
            self.X, self.y = _synth_regression_dataset(n_samples=1000000,
                                                       n_features=500)
        else:
            self.X, self.y = _20newsgroups_highdim_dataset(ngrams=(1, 5))

        self.ridge_params = {'solver': 'lsqr',
                             'fit_intercept': False,
                             'random_state': 0}

    def time_fit(self, *args):
        ridge = Ridge(**self.ridge_params)
        ridge.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        ridge = Ridge(**self.ridge_params)
        ridge.fit(self.X, self.y)


class LinearRegression_bench(Benchmark):
    """
    Benchmarks for Linear Reagression.
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        if representation is 'dense':
            self.X, self.y = _synth_regression_dataset(n_samples=1000000,
                                                       n_features=100)
        else:
            self.X, self.y = _20newsgroups_highdim_dataset(n_samples=5000)

    def time_fit(self, *args):
        linear_reg = LinearRegression()
        linear_reg.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        linear_reg = LinearRegression()
        linear_reg.fit(self.X, self.y)


class SGDRegressor_bench(Benchmark):
    """
    Benchmark for SGD
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        if representation is 'dense':
            self.X, self.y = _synth_regression_dataset(n_samples=100000,
                                                       n_features=100)
        else:
            self.X, self.y = _20newsgroups_highdim_dataset(n_samples=5000)

        self.sgdr_params = {'max_iter': 1000,
                            'tol': 1e-16}

    def time_fit(self, *args):
        sgd_reg = SGDRegressor(**self.sgdr_params)
        sgd_reg.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        sgd_reg = SGDRegressor(**self.sgdr_params)
        sgd_reg.fit(self.X, self.y)


class ElasticNet_bench(Benchmark):
    """
    Benchmarks for ElasticNet.
    """

    param_names = ['representation', 'precompute']
    params = (['dense', 'sparse'], [True, False])

    def setup(self, *params):
        representation, precompute = params

        if representation is 'dense':
            self.X, self.y = _synth_regression_dataset(n_samples=1000000,
                                                       n_features=100)
        else:
            self.X, self.y = _20newsgroups_highdim_dataset()

        self.en_params = {'precompute': precompute,
                          'random_state': 0}

    def time_fit(self, *args):
        en = ElasticNet(**self.en_params)
        en.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        en = ElasticNet(**self.en_params)
        en.fit(self.X, self.y)


class Lasso_bench(Benchmark):
    """
    Benchmarks for Lasso.
    """

    param_names = ['representation', 'precompute']
    params = (['dense', 'sparse'], [True, False])

    def setup(self, *params):
        representation, precompute = params

        if representation is 'dense':
            self.X, self.y = _synth_regression_dataset(n_samples=1000000,
                                                       n_features=100)
        else:
            self.X, self.y = _20newsgroups_highdim_dataset()

        self.lasso_params = {'precompute': precompute,
                             'random_state': 0}

    def time_fit(self, *args):
        lasso = Lasso(**self.lasso_params)
        lasso.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        lasso = Lasso(**self.lasso_params)
        lasso.fit(self.X, self.y)
