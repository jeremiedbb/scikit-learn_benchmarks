from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .common import Benchmark
from .datasets import (_20newsgroups_highdim_dataset,
                       _20newsgroups_lowdim_dataset)


class RandomForestClassifier_bench(Benchmark):
    """
    Benchmarks for RandomForestClassifier.
    """
    # params = (representation)
    param_names = ['params'] + Benchmark.param_names
    params = ([('dense', ),
               ('sparse', )], ) + Benchmark.params

    def setup(self, params, *common):
        representation = params[0]

        n_jobs = common[0]

        if representation is 'sparse':
            self.X, self.y = _20newsgroups_highdim_dataset()
        else:
            self.X, self.y = _20newsgroups_lowdim_dataset()

        self.rf_params = {'n_estimators': 100,
                          'min_samples_split': 10,
                          'max_features': 'log2',
                          'n_jobs': n_jobs,
                          'random_state': 0}

    def time_fit(self, *args):
        rf = RandomForestClassifier(**self.rf_params)
        rf.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        rf = RandomForestClassifier(**self.rf_params)
        rf.fit(self.X, self.y)


class GradientBoostingClassifier_bench(Benchmark):
    """
    Benchmarks for GradientBoostingClassifier.
    """
    # params = (representation)
    param_names = ['params']
    params = ([('dense', ),
               ('sparse', )], )

    def setup(self, params, *common):
        representation = params[0]

        if representation is 'sparse':
            self.X, self.y = _20newsgroups_highdim_dataset()
        else:
            self.X, self.y = _20newsgroups_lowdim_dataset()

        self.gb_params = {'n_estimators': 10,
                          'max_features': 'log2',
                          'subsample': 0.5,
                          'random_state': 0}

    def time_fit(self, *args):
        gb = GradientBoostingClassifier(**self.gb_params)
        gb.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        gb = GradientBoostingClassifier(**self.gb_params)
        gb.fit(self.X, self.y)
