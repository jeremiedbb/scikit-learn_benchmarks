from sklearn.linear_model import (LogisticRegression, Ridge, ElasticNet, Lasso,
                                  LinearRegression, SGDRegressor)

from .common import Benchmark, Estimator_bench, Predictor_bench
from .datasets import (_20newsgroups_highdim_dataset,
                       _20newsgroups_lowdim_dataset, _synth_regression_dataset)


class LogisticRegression_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmarks for LogisticRegression.
    """

    param_names = ['representation', 'solver', 'n_jobs']
    params = (['dense', 'sparse'], ['lbfgs', 'saga'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        representation, solver, n_jobs = params

        penalty = 'l2' if solver == 'lbfgs' else 'l1'

        if Benchmark.data_size == 'large':
            if representation == 'sparse':
                data = _20newsgroups_highdim_dataset(n_samples=10000)
            else:
                data = _20newsgroups_lowdim_dataset(n_components=1e3)
        else:
            if representation == 'sparse':
                data = _20newsgroups_highdim_dataset(n_samples=2500)
            else:
                data = _20newsgroups_lowdim_dataset()
        self.X, _, self.y, _ = data

        self.estimator = LogisticRegression(solver=solver,
                                            penalty=penalty,
                                            multi_class='multinomial',
                                            tol=0.01,
                                            n_jobs=n_jobs,
                                            random_state=0)

        self.estimator.fit(self.X, self.y)


class Ridge_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmarks for Ridge.
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        if representation == 'dense':
            data = _synth_regression_dataset(n_samples=100000, n_features=500)
        else:
            data = _20newsgroups_highdim_dataset(ngrams=(1, 3))
        self.X, _, self.y, _ = data

        self.estimator = Ridge(solver='saga',
                               fit_intercept=False,
                               random_state=0)

        self.estimator.fit(self.X, self.y)


class LinearRegression_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmarks for Linear Reagression.
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        if representation == 'dense':
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _20newsgroups_highdim_dataset(n_samples=5000)
        self.X, _, self.y, _ = data

        self.estimator = LinearRegression()

        self.estimator.fit(self.X, self.y)


class SGDRegressor_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmark for SGD
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        if representation == 'dense':
            data = _synth_regression_dataset(n_samples=100000, n_features=100)
        else:
            data = _20newsgroups_highdim_dataset(n_samples=5000)
        self.X, _, self.y, _ = data

        self.estimator = SGDRegressor(max_iter=1000,
                                      tol=1e-16)

        self.estimator.fit(self.X, self.y)


class ElasticNet_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmarks for ElasticNet.
    """

    param_names = ['representation', 'precompute']
    params = (['dense', 'sparse'], [True, False])

    def setup(self, *params):
        representation, precompute = params

        if representation == 'dense':
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _20newsgroups_highdim_dataset()
        self.X, _, self.y, _ = data

        self.estimator = ElasticNet(precompute=precompute,
                                    random_state=0)

        self.estimator.fit(self.X, self.y)


class Lasso_bench(Benchmark, Estimator_bench, Predictor_bench):
    """
    Benchmarks for Lasso.
    """

    param_names = ['representation', 'precompute']
    params = (['dense', 'sparse'], [True, False])

    def setup(self, *params):
        representation, precompute = params

        if representation == 'dense':
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _20newsgroups_highdim_dataset()
        self.X, _, self.y, _ = data

        self.lasso_params = {'precompute': precompute,
                             'random_state': 0}
        self.estimator = Lasso(precompute=precompute,
                               random_state=0)

        self.estimator.fit(self.X, self.y)
