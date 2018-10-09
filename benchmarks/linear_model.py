from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, Lasso

from .common import Benchmark
from .datasets import (_20newsgroups_highdim_dataset,
                       _20newsgroups_lowdim_dataset,
                       _synth_regression_lowdim_dataset,
                       _synth_regression_highdim_dataset)


class LogisticRegression_bench(Benchmark):
    """
    Benchmarks for LogisticRegression.
    """
    # params = (representation, solver)
    param_names = ['params'] + Benchmark.param_names
    params = ([('dense', 'lbfgs'),
               ('dense', 'saga'),
               ('sparse', 'lbfgs'),
               ('sparse', 'saga')],) + Benchmark.params

    def setup(self, params, *common):
        representation = params[0]
        solver = params[1]

        n_jobs = common[0]

        if representation is 'sparse':
            self.X, self.y = _20newsgroups_highdim_dataset()
        else:
            self.X, self.y = _20newsgroups_lowdim_dataset()

        if solver is 'lbfgs':
            self.lr_params = {'penalty': 'l2'}
        else:
            self.lr_params = {'penalty': 'l1'}

        self.lr_params.update({'solver': solver,
                               'multi_class': 'multinomial',
                               'tol': 0.01,
                               'max_iter': 300,
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
    # params = (representation)
    param_names = ['params']
    params = ([('dense',),
               ('sparse',)],)

    def setup(self, params):
        representation = params[0]

        if representation is 'sparse':
            self.X, self.y = _synth_regression_highdim_dataset()
        else:
            self.X, self.y = _synth_regression_lowdim_dataset()

        self.ridge_params = {'solver': 'lsqr',
                             'fit_intercept': False,
                             'random_state': 0}

    def time_fit(self, *args):
        ridge = Ridge(**self.ridge_params)
        ridge.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        ridge = Ridge(**self.ridge_params)
        ridge.fit(self.X, self.y)


class ElasticNet_bench(Benchmark):
    """
    Benchmarks for ElasticNet.
    """
    # params = (representation, precompute)
    param_names = ['params']
    params = ([('dense', True),
               ('dense', False),
               ('sparse', True),
               ('sparse', False)],)

    def setup(self, params):
        representation = params[0]
        precompute = params[1]

        if representation is 'sparse':
            self.X, self.y = _synth_regression_highdim_dataset()
        else:
            self.X, self.y = _synth_regression_lowdim_dataset()

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
    # params = (representation, precompute)
    param_names = ['params']
    params = ([('dense', True),
               ('dense', False),
               ('sparse', True),
               ('sparse', False)],)

    def setup(self, params):
        representation = params[0]
        precompute = params[1]

        if representation is 'sparse':
            self.X, self.y = _synth_regression_highdim_dataset()
        else:
            self.X, self.y = _synth_regression_lowdim_dataset()

        self.lasso_params = {'precompute': precompute,
                             'random_state': 0}

    def time_fit(self, *args):
        lasso = Lasso(**self.lasso_params)
        lasso.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        lasso = Lasso(**self.lasso_params)
        lasso.fit(self.X, self.y)
