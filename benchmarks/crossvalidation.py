from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from .common import Benchmark

from .datasets import _synth_classification_dataset


class CrossValidation_bench(Benchmark):
    def setup(self, params):

        self.n_jobs = params[0]
        self.X, self.y = _synth_classification_dataset(
            n_samples=70000, n_features=200, random_state=0
        )

    def time_crossval(self, params):
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        cross_val_score(
            clf, self.X, self.y, n_jobs=self.n_jobs, cv=4
        )

    def time_gridsearch(self, params):
        # Setup for GridSearchCV
        gs_parameters = {
            "n_estimators": [10, 50, 100, 500],
            "max_depth": [5, 10, None],
            "max_features": [0.1, 0.5, 0.8, 1.0],
        }
        gs_clf = GridSearchCV(
            RandomForestClassifier(), gs_parameters, cv=3, n_jobs=self.n_jobs
        )
        gs_clf.fit(self.X, self.y)
