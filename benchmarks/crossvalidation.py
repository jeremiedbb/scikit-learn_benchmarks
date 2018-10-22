from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from .common import Benchmark

from .datasets import _synth_classification_dataset


class CrossValidationSuite(Benchmark):
    def setup(self, params):
        X, y = _synth_classification_dataset(
            n_samples=70000, n_features=200, random_state=0
        )

        self.clf = RandomForestClassifier(n_estimators=100, random_state=0)

        # Setup for GridSearchCV
        parameters = {
            "n_estimators": [10, 50, 100, 500],
            "max_depth": [5, 10, None],
            "max_features": [0.1, 0.5, 0.8, 1.0],
        }
        self.gs_clf = GridSearchCV(
            RandomForestClassifier(), parameters, cv=3, n_jobs=params
        )

    def time_crossval(self, params):
        cross_val_score(self.clf, self.X_train, self.y_train, n_jobs=params, cv=4)

    def time_gridsearch(self):
        self.gs_clf.fit(self.X_train, self.y_train)
