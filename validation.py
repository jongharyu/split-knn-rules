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

    def grid_search(self, X, y):
        # search for an optimal k and fit
        # 1) coarse search: find best k in [1,2,4,8,16,...]
        k_set = []
        k_err = []
        k = 1
        while k < np.sqrt(X.shape[0]):
            k_set.append(k)
            k_err.append(self.cross_validate(X, y, k))
            k = k * 2
        k_opt_rough = k_set[np.argmin(k_err)]

        # 2) fine search: find best k in [.5 * k_opt_rough - 10, 2 * k_opt_rough + 10]
        k_search_start = (max(1, int(.5 * k_opt_rough) - 10) // 2) * 2 + 1
        k_search_end = int(min(2 * k_opt_rough + 11, np.sqrt(X.shape[0])))
        for k in range(k_search_start, k_search_end, 2):
            if k not in k_set:
                k_set.append(k)
                k_err.append(self.cross_validate(X, y, k))
        k_opt = k_set[np.argmin(k_err)]

        return k_opt


class GridSearchWithCrossValidationForSplitSelect1NN:
    def __init__(self, n_folds=5, n_repeat=1, parallel=False):
        self.n_folds = n_folds
        self.n_repeat = n_repeat
        self.parallel = parallel
        
    def cross_validate(self, X, y, n_splits, select_ratio):
        # calculate error rate of a given k through cross validation
        errors = []
        for repeat in range(self.n_repeat):
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)

            for train_index, valid_index in skf.split(X, y):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]

                regressor = SplitSelectKNeighborsRegressor(n_neighbors=1, n_splits=n_splits, select_ratio=select_ratio, verbose=False)
                regressor.fit(X_train, y_train)
                y_valid_pred = regressor.predict(X_valid, parallel=self.parallel)['soft1_select1_1NN'] > .5
                errors.append(compute_error_rate(y_valid, y_valid_pred))

        return np.mean(errors)

    def grid_search(self, X, y):
        # search for an optimal k and fit
        # 1) coarse search: find best k in [1,2,4,8,16,...]
        n_splits_set = []
        n_splits_err = []
        n_splits = 2
        while n_splits < np.sqrt(X.shape[0]):
            n_splits_set.append(n_splits)
            n_splits_err.append(self.cross_validate(X, y, n_splits, select_ratio=None))
            n_splits = n_splits * 2
        n_splits_opt_rough = n_splits_set[np.argmin(n_splits_err)]

        # 2) fine search: find best k in [.5 * k_opt_rough - 10, 2 * k_opt_rough + 10]
        n_splits_search_start = (max(2, int(.5 * n_splits_opt_rough) - 10) // 2) * 2 + 1
        n_splits_search_end = int(min(2 * n_splits_opt_rough + 11, np.sqrt(X.shape[0])))
        for n_splits in range(n_splits_search_start, n_splits_search_end, 2):
            if n_splits not in n_splits_set:
                n_splits_set.append(n_splits)
                n_splits_err.append(self.cross_validate(X, y, n_splits, select_ratio=None))
        n_splits_opt = n_splits_set[np.argmin(n_splits_err)]

        # 3) find best select_ratio in np.linspace(1-exp(-1/4), 1, 10)
        select_ratio_set = []
        select_ratio_err = []
        select_ratio_start = 1 - np.exp(-1 / 4)
        for select_ratio in np.linspace(select_ratio_start, 0.9, 10):
            select_ratio_set.append(select_ratio)
            select_ratio_err.append(self.cross_validate(X, y, n_splits_opt, select_ratio=select_ratio))
        select_ratio_opt = select_ratio_set[np.argmin(select_ratio_err)]

        return n_splits_opt, select_ratio_opt