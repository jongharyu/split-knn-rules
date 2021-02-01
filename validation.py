import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from regressor import SplitSelectKNeighborsRegressor


def compute_error_rate(truth, prediction):
    wrong = 0
    for x, y in zip(truth, prediction):
        if x != y:
            wrong = wrong + 1
    return wrong / len(truth)


class GridSearchWithCrossValidationForKNeighborsClassifier:
    # Reference: https://github.com/lirongx/SubNN/blob/master/SubNN.py
    def __init__(self, n_folds=5, n_repeat=1):
        self.n_folds = n_folds
        self.n_repeat = n_repeat

    def cross_validate(self, X, y, k):
        # calculate error rate of a given k through cross validation
        errors = []
        for repeat in range(self.n_repeat):
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)

            for train_index, valid_index in skf.split(X, y):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]

                if k > len(y_train):
                    k = len(y_train)
                classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(X_train, y_train)
                y_pred = classifier.predict(X_valid)
                errors.append(compute_error_rate(y_valid, y_pred))

        return np.mean(errors)

    def grid_search(self, X, y, max_k=None, fine_search=False):
        # search for an optimal k and fit
        if not max_k:
            max_k = X.shape[0]

        # 1) coarse search: find best k in [3,7,15,31,...]
        k_set = []
        k_err = []
        k = 3
        while k < max_k:
            k_set.append(k)
            k_err.append(self.cross_validate(X, y, k))
            k = 2 * (k + 1) - 1
        k_opt_rough = k_set[np.argmin(k_err)]

        if fine_search:
            # 2) fine search: find best k in [.5 * k_opt_rough - 10, 2 * k_opt_rough + 10]
            k_search_start = np.max([(max(1, int(.5 * k_opt_rough) - 10) // 2) * 2 + 1, 3])
            k_search_end = int(min(2 * k_opt_rough + 11, np.sqrt(X.shape[0])))
            for k in range(k_search_start, k_search_end, 2):
                if k not in k_set:
                    k_set.append(k)
                    k_err.append(self.cross_validate(X, y, k))
        k_set = np.array(k_set)
        k_err = np.array(k_err)
        k_opt = k_set[np.argmin(k_err)]
        indices = np.argsort(k_set)
        profile = np.vstack([k_set[indices], k_err[indices]])  # (2, len(k_set))
        return k_opt, profile


class GridSearchWithCrossValidationForSplitSelect1NN(GridSearchWithCrossValidationForKNeighborsClassifier):
    def __init__(self, parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.parallel = parallel
        
    def cross_validate(self, X, y, n_splits):
        # calculate error rate of a given k through cross validation
        errors = []
        for repeat in range(self.n_repeat):
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)

            for train_index, valid_index in skf.split(X, y):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]

                regressor = SplitSelectKNeighborsRegressor(n_neighbors=1, n_splits=n_splits, select_ratio=None, verbose=False)
                regressor.fit(X_train, y_train)
                y_valid_pred = regressor.predict(X_valid, parallel=self.parallel)['split_select1_1NN'] > .5
                errors.append(compute_error_rate(y_valid, y_valid_pred))

        return np.mean(errors)
