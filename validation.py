import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from regressor import SplitSelectKNeighborsRegressor


def compute_error(truth, prediction, classification=True):
    if classification:
        wrong = 0
        for x, y in zip(truth, prediction):
            if x != y:
                wrong = wrong + 1
        return wrong / len(truth)
    else:
        # For regression, l1-norm
        assert len(truth.shape) in [1, 2]
        if len(truth.shape) == 1:
            return np.abs(truth - prediction).mean()
        else:
            return np.abs(truth - prediction).sum(axis=1).mean()


class GridSearchForKNeighborsEstimator:
    # Reference: https://github.com/lirongx/SubNN/blob/master/SubNN.py
    def __init__(self, n_folds=5, n_repeat=1, verbose=True, max_valid_size=1000, classification=True):
        self.n_folds = n_folds
        self.n_repeat = n_repeat
        self.verbose = verbose
        self.max_valid_size = max_valid_size
        self.classification = classification

    def compute_error(self, X_train, y_train, X_valid, y_valid, k, **kwargs):
        if k > len(y_train):
            k = len(y_train)
        Estimator = KNeighborsClassifier if self.classification else KNeighborsRegressor
        estimator = Estimator(
            n_neighbors=k,
            n_jobs=-1,
        ).fit(X_train, y_train)
        y_pred = estimator.predict(X_valid)
        error = compute_error(y_valid, y_pred, self.classification)
        return error

    def cross_validate(self, X, y, k, **kwargs):
        # calculate error rate of a given k through cross validation
        errors = []
        for repeat in range(self.n_repeat):
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
            for train_index, valid_index in skf.split(X, y):
                X_train, X_valid = X[train_index], X[valid_index[:self.max_valid_size]]
                y_train, y_valid = y[train_index], y[valid_index[:self.max_valid_size]]
                error = self.compute_error(X_train, y_train, X_valid, y_valid, k, **kwargs)
                errors.append(error)

        return np.mean(errors)

    def grid_search(self, X, y, k_max=None, fine_search=False):
        # search for an optimal k
        if not k_max:
            k_max = X.shape[0]

        if self.verbose:
            print("\t\tValidating (max={}): ".format(int(k_max)), end='')

        # 1) coarse search: find best k in [3, 7, 15, 31,...]
        k_set = []
        k_err = []
        k = 3
        while k < k_max:
            k_set.append(k)
            if self.verbose:
                print(k, end=' ')
            err = self.cross_validate(X, y, k)
            k_err.append(err)
            k = 2 * (k + 1) - 1
        k_opt_rough = k_set[np.argmin(k_err)]

        # 2) (optional) fine search: find best k in [.5 * k_opt_rough - 10, 2 * k_opt_rough + 10]
        if fine_search:
            k_search_start = np.max([(max(1, int(.5 * k_opt_rough) - 10) // 2) * 2 + 1, 3])
            k_search_end = int(min(2 * k_opt_rough + 11, np.sqrt(X.shape[0])))
            for k in range(k_search_start, k_search_end, 2):
                if k not in k_set:
                    k_set.append(k)
                    if self.verbose:
                        print(k, end=' ')
                    err = self.cross_validate(X, y, k)
                    k_err.append(err)

        k_set = np.array(k_set)
        k_err = np.array(k_err)
        k_opt = k_set[np.argmin(k_err)]
        indices = np.argsort(k_set)
        profile = np.vstack([k_set[indices], k_err[indices]])  # (2, len(k_set))
        if self.verbose:
            print()
        return k_opt, profile


class GridSearchForSplitSelect1NeighborEstimator(GridSearchForKNeighborsEstimator):
    def __init__(self, parallel=False, classification=True, onehot_encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.parallel = parallel
        self.classification = classification
        self.onehot_encoder = onehot_encoder

    def compute_error(self, X_train, y_train, X_valid, y_valid, k, select_ratio=None):
        estimator = SplitSelectKNeighborsRegressor(
            n_neighbors=1,
            n_splits=k,
            select_ratio=select_ratio,
            verbose=False,
            classification=self.classification,
            onehot_encoder=self.onehot_encoder).fit(X_train, y_train)
        y_pred = estimator.predict(X_valid, parallel=self.parallel)['split_select1_1NN']
        error = compute_error(y_valid, y_pred, self.classification)
        return error

    def grid_search(self, X, y, k_max=None, fine_search=False, search_select_ratio=False):
        # search for an optimal k (and optionally an optimal select ratio)
        n_split_opt, n_split_profile = super().grid_search(X, y, k_max, fine_search)
        select_ratio_opt = None
        select_ratio_profile = None
        if search_select_ratio:
            if self.verbose:
                print("\t\tValidating select ratio: ", end='')
            # find best select_ratio in [.1, .2, ..., .9]
            param_set = []
            param_err = []
            default_select_ratio = 1 - np.exp(-1 / 4)
            for select_ratio in [i * default_select_ratio / 2 for i in range(1, 9)]:
                param_set.append(select_ratio)
                if self.verbose:
                    print('{:.2f}'.format(select_ratio), end=' ')
                err = self.cross_validate(X, y, n_split_opt, select_ratio=select_ratio)
                param_err.append(err)
            select_ratio_opt = param_set[np.argmin(param_err)]
            select_ratio_profile = np.vstack([param_set, param_err])  # (2, len(param_set))
            if self.verbose:
                print()

        return n_split_opt, n_split_profile, select_ratio_opt, select_ratio_profile
