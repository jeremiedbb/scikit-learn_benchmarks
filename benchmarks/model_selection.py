from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from .common import Benchmark, Estimator_bench
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

        self.X, _, self.y, _ = _synth_classification_dataset(n_samples=50000,
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


class GridSearch_bench(Benchmark, Estimator_bench):
    """
    Benchmarks for GridSearch.
    """

    timeout = 20000

    param_names = ['n_jobs']
    params = (Benchmark.n_jobs_vals,)

    def setup(self, *params):
        n_jobs, = params

        self.X, _, self.y, _ = _synth_classification_dataset(n_samples=10000,
                                                             n_features=100)

        clf = RandomForestClassifier(random_state=0)

        if Benchmark.data_size == 'large':
            n_estimators_list = [10, 25, 50, 100, 500]
            max_depth_list = [5, 10, None]
            max_features_list = [0.1, 0.4, 0.8, 1.0]
        else:
            n_estimators_list = [10, 25, 50]
            max_depth_list = [5, 10]
            max_features_list = [0.1, 0.4, 0.8]

        param_grid = {'n_estimators': n_estimators_list,
                      'max_depth': max_depth_list,
                      'max_features': max_features_list}

        self.estimator = GridSearchCV(clf, param_grid, n_jobs=n_jobs, cv=4)

        self.estimator.fit(self.X, self.y)
