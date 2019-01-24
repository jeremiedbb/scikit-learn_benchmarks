from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .common import Benchmark, Predictor
from .datasets import (_20newsgroups_highdim_dataset,
                       _20newsgroups_lowdim_dataset)


class RandomForestClassifier_bench(Benchmark, Predictor):
    """
    Benchmarks for RandomForestClassifier.
    """

    param_names = ['representation', 'n_jobs']
    params = (['dense', 'sparse'], Benchmark.n_jobs_vals)

    def setup(self, *params):
        representation, n_jobs = params

        n_estimators = 500 if Benchmark.data_size == 'large' else 100

        if representation == 'sparse':
            data = _20newsgroups_highdim_dataset()
        else:
            data = _20newsgroups_lowdim_dataset()
        self.X, _, self.y, self.y_val = data

        self.estimator = RandomForestClassifier(n_estimators=n_estimators,
                                                min_samples_split=10,
                                                max_features='log2',
                                                n_jobs=n_jobs,
                                                random_state=0)

        self.estimator.fit(self.X, self.y)


class GradientBoostingClassifier_bench(Benchmark, Predictor):
    """
    Benchmarks for GradientBoostingClassifier.
    """

    param_names = ['representation']
    params = (['dense', 'sparse'],)

    def setup(self, *params):
        representation, = params

        n_estimators = 100 if Benchmark.data_size == 'large' else 10

        if representation == 'sparse':
            data = _20newsgroups_highdim_dataset()
        else:
            data = _20newsgroups_lowdim_dataset()
        self.X, _, self.y, self.y_val = data

        self.estimator = GradientBoostingClassifier(n_estimators=n_estimators,
                                                    max_features='log2',
                                                    subsample=0.5,
                                                    random_state=0)

        self.estimator.fit(self.X, self.y)
