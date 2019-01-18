from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from .common import Benchmark
from .datasets import _synth_classification_dataset


class CrossValidation_bench(Benchmark):
    """
    Benchmarks for Cross Validation.
    """

    timeout = 20000

    param_names = ['n_jobs']
    params = (Benchmark.n_jobs_vals,)

    def setup(self, *params):
        n_jobs, = params

        self.X, self.y = _synth_classification_dataset(n_samples=50000,
                                                       n_features=100)

        self.clf = RandomForestClassifier(n_estimators=50,
                                          max_depth=10,
                                          random_state=0)

        cv = 16 if Benchmark.data_size == 'large' else 4

        self.cv_params = {'n_jobs': n_jobs,
                          'cv': cv}

    def time_crossval(self, *args):
        cross_val_score(self.clf, self.X, self.y, **self.cv_params)

    def peakmem_fit(self, *args):
        cross_val_score(self.clf, self.X, self.y, **self.cv_params)


class GridSearch_bench(Benchmark):
    """
    Benchmarks for GridSearch.
    """

    timeout = 20000

    param_names = ['n_jobs']
    params = (Benchmark.n_jobs_vals,)

    def setup(self, *params):
        n_jobs, = params

        self.X, self.y = _synth_classification_dataset(n_samples=10000,
                                                       n_features=100)

        self.clf = RandomForestClassifier(random_state=0)

        self.param_grid = {'n_estimators': [10, 25, 50],
                           'max_depth': [5, 10],
                           'max_features': [0.1, 0.4, 0.8]}

        if Benchmark.data_size == 'large':
            self.param_grid['n_estimators'].append([100, 500])
            self.param_grid['max_depth'].append(None)
            self.param_grid['max_features'].append(1.0)

        self.gs_params = {'n_jobs': n_jobs,
                          'cv': 4}

    def time_fit(self, *args):
        gs_clf = GridSearchCV(self.clf, self.param_grid, **self.gs_params)
        gs_clf.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        gs_clf = GridSearchCV(self.clf, self.param_grid, **self.gs_params)
        gs_clf.fit(self.X, self.y)
