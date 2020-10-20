import multiprocessing as mp
from timeit import default_timer as timer

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def compute_error_rate(truth, prediction):
    wrong = 0
    for x, y in zip(truth, prediction):
        if x != y:
            wrong = wrong + 1
    return wrong / len(truth)


from sklearn.neighbors._base import _get_weights
from sklearn.utils import check_array


class ModifiedKNeighborsRegressor(KNeighborsRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X, k=None):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        k: for compatibility in local_predict of SplitKNeighbor

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred, neigh_dist[:, -1]  # return the k-th NN distances


class KNeighborsClassifierWithCrossValidation(KNeighborsClassifier):
    # Reference: https://github.com/lirongx/SubNN/blob/master/SubNN.py
    def __init__(self, *args, **kwargs):
        super(self).__init__(*args, **kwargs)

    # calculate error rate of a given k through cross validation
    def cross_validate_k(self, X, y, k, n_folds=2, n_repeat=1):
        scores = []
        for repeat in range(n_repeat):
            skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)

            for train_index, test_index in skf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if k > len(y_train):
                    k = len(y_train)
                self.k_train = k
                self.fit(X_train, y_train)
                y_pred = self.predict(X_test)
                scores.append(compute_error_rate(y_test, y_pred))

        return np.mean(scores)

    # search for a optimal k and fit
    def search_k(self, X, y, n_folds=2, n_repeat=1):
        # search best k in 1,2,4,8,16,...
        k_set = []
        k_err = []
        k = 1
        while k < X.shape[0]:
            k_set.append(k)
            k_err.append(self.cross_validate_k(X, y, k, n_folds=n_folds, n_repeat=n_repeat))
            k = k * 2
        k_opt_rough = k_set[np.argmin(k_err)]

        # search for optimal k in [k_opt_rough/2, k_opt_rough*2]
        for k in range(max(1, int(k_opt_rough / 2) - 10), min(k_opt_rough * 2 + 11, X.shape[0])):
            if k not in k_set:
                k_set.append(k)
                k_err.append(self.cross_validate_k(X, y, k, n_folds=n_folds, n_repeat=n_repeat))

        k_opt = k_set[np.argmin(k_err)]
        self.k_train = k_opt
        return k_opt


class SplitKNeighbor:
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None,
                 kappa=0.9,
                 **kwargs, ):
        # algorithm: one of {'auto', 'ball_tree', 'kd_tree', 'brute'}
        self.n_neighbors = n_neighbors
        self.base_kwargs = {'weights': weights,
                            'algorithm': algorithm,
                            'leaf_size': leaf_size,
                            'p': p,
                            'metric': metric,
                            'metric_params': metric_params,
                            **kwargs}

        self.kappa = kappa
        self.sigma = 1 - np.exp(- self.n_neighbors / 4)
        self._fit_method = None
        self.base_model = None

    def fit(self, *data_split):
        # TODO: check if classifier.predict_proba is equivalent to regressor.predict for default cases
        self.local_models = [self.base_model(n_neighbors=self.n_neighbors, **self.base_kwargs).fit(*split)
                             for i, split in enumerate(zip(*data_split))]
        self._fit_method = self.local_models[0]._fit_method

        self.n_splits = len(self.local_models)
        n_selected = int(np.floor(self.kappa * self.n_splits * self.sigma))  # number of estimates to be selected
        self.n_selected = max([n_selected, 1])

        return self

    def local_predict(self, X, k=None, parallel=False):
        # k is None if and only if local model is a regressor
        n_k = 1 if k is None else len(k)

        n_queries = X.shape[0]
        local_estimates = np.zeros((self.n_splits, n_queries, n_k))  # local estimates
        knn_distances = np.zeros((self.n_splits, n_queries, n_k))  # kNN distances to the query
        if n_k == 1:
            local_estimates = local_estimates.squeeze(-1)
            knn_distances = knn_distances.squeeze(-1)

        # Local kNN operations
        start = timer()
        if not parallel:
            for m, model in enumerate(self.local_models):
                local_estimates[m], knn_distances[m] = model.predict(X, k=k)  # (n_queries, n_k) or (n_queries,)
        else:  # parallel processing
            with mp.Pool() as pool:
                local_returns = pool.map(predict, zip(self.local_models, [X] * self.n_splits))
            local_estimates, knn_distances = [np.array(l) for l in list(zip(*local_returns))]
        elapsed_time = timer() - start
        print("\n\tLocal kNN operations: {:.2f}s / {:.4f}s (per split)".format(
            elapsed_time, elapsed_time / self.n_splits)
        )
        return local_estimates, knn_distances

    @staticmethod
    def get_random_split(data, n_splits):
        # assume data is list
        if not isinstance(data, list):
            data = [data]
        N = len(data[0])
        rand_perm = np.random.permutation(N)
        data = [x[rand_perm] for x in data]  # shuffling data
        return [np.array_split(x, n_splits) for x in data]


class SplitKNeighborsRegressor(SplitKNeighbor):
    def __init__(self, thresholding=False, **kwargs):
        super().__init__(**kwargs)
        self.thresholding = thresholding  # if True, take thresholding after each local estimates
        self.local_models = []
        self.base_model = ModifiedKNeighborsRegressor

    def predict(self, X, parallel=False):
        # X: np.array; (n_queries, n_features)
        n_queries = X.shape[0]

        # Local operations
        local_estimates, knn_distances = self.local_predict(X, parallel=parallel)  # (n_samples, n_queries)

        # Global aggregation
        # Note that knn_distances.shape = (n_splits, n_queries)
        start = timer()
        selected_indices = np.argpartition(knn_distances, self.n_selected, axis=0)[:self.n_selected, :]  # (n_selected, n_queries); takes O(n_splits)
        print("\tFind L distances: {:.4f}s".format(timer() - start))

        final_estimates = dict()
        # start = timer()
        final_estimates['soft1_selective1_{}NN'.format(self.n_neighbors)] = \
            local_estimates[
            selected_indices,
            np.repeat(np.arange(n_queries).reshape(1, n_queries), self.n_selected, axis=0)
        ].mean(axis=0)  # (n_queries)
        # print("Select and compute mean: {:.4f}s".format(elapsed_time, timer() - start))
        final_estimates['soft0_selective1_{}NN'.format(self.n_neighbors)] = \
            (local_estimates > 0.5)[
            selected_indices,
            np.repeat(np.arange(n_queries).reshape(1, n_queries), self.n_selected, axis=0)
        ].mean(axis=0)  # (n_queries)
        final_estimates['soft1_selective0_{}NN'.format(self.n_neighbors)] = local_estimates.mean(axis=0)  # (n_queries,)
        final_estimates['soft0_selective0_{}NN'.format(self.n_neighbors)] = (local_estimates > 0.5).mean(axis=0)  # (n_queries,)

        return final_estimates


# for multiprocessing
def predict(model_data):
    model, data = model_data
    return model.predict(data)
