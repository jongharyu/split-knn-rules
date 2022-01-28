from timeit import default_timer as timer

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights
from sklearn.utils import check_array


class ExtendedKNeighborsRegressor(KNeighborsRegressor):
    """
    A modified version of the scikit implementation of k-NN regression algorithm.
    It is modified so that
        1) it returns k-NN distances along with the regression estimates and
        2) it support multidimensional regression.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
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
            # y_pred = np.mean(_y[neigh_ind], axis=1)  # modified as below
            # in case of multilabel classification / multidimensional regression
            if hasattr(self._y, 'toarray'):
                y_pred = self._y[neigh_ind.reshape((-1,))].toarray().reshape((*neigh_ind.shape, -1)).mean(axis=1)
            else:
                y_pred = self._y[neigh_ind.reshape((-1,))].reshape((*neigh_ind.shape, -1)).mean(axis=1)
        else:
            # TODO: code below may not work for multidimensional target
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred, neigh_dist[:, -1]  # return the k-th NN distances


class SplitSelectKNeighbors:
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None,
                 n_splits=1,
                 n_select=None,
                 select_ratio=None,
                 verbose=True,
                 onehot_encoder=None,
                 classification=False,
                 density=False,
                 pool=None,
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

        self._fit_method = None
        self.base_model = None
        self.n_splits = n_splits
        self.select_ratio = 1 - np.exp(- self.n_neighbors / 4) if not select_ratio else select_ratio
        if not n_select:
            n_select = n_splits
            if self.select_ratio < 1:
                # If n_select is not specified (None), use the default parameter from theoretical guarantee
                n_select = int(np.ceil(self.n_splits * self.select_ratio))  # number of estimates to be selected
                n_select = min([max([n_select, 1]), n_splits - 1])  # CRUCIAL! Otherwise, argpartition raises an unknown error
        # If n_select is specified, select_ratio is ignored
        self.n_select = n_select

        self.is_classifier = classification
        self.is_density_estimator = density
        self.onehot_encoder = onehot_encoder  # in case of multilabel classification
        self.verbose =verbose
        self.target_dim = None
        self.pool = pool

    @property
    def multilabel_classification(self):
        return self.onehot_encoder

    @property
    def n_classes(self):
        if not self.is_classifier:
            return -1
        else:
            if not self.multilabel_classification:
                return 2
            else:
                return self.onehot_encoder.categories_[0].size

    def fit(self, *data_train):
        # TODO: check if classifier.predict_proba is equivalent to regressor.predict for default cases
        data_train = list(data_train)
        if self.is_density_estimator:
            self.target_dim = 1
        elif self.multilabel_classification:
            data_train[1] = self.onehot_encoder.transform(data_train[1].reshape((-1, 1))).toarray()
            self.target_dim = data_train[1].shape[1]
        elif len(data_train[1].shape) == 1:
            self.target_dim = 1
        else:
            # multidimensional regression
            self.target_dim = data_train[1].shape[1]

        data_split = self.get_random_split(data_train, self.n_splits)
        self.local_models = [self.base_model(n_neighbors=self.n_neighbors, **self.base_kwargs).fit(*split)
                             for i, split in enumerate(zip(*data_split))]
        self._fit_method = self.local_models[0]._fit_method

        return self

    def local_predict(self, X, parallel=False):
        n_queries = X.shape[0]
        local_estimates = np.zeros((self.n_splits, n_queries, self.target_dim))  # local estimates
        knn_distances = np.zeros((self.n_splits, n_queries))  # kNN distances to the query
        if self.target_dim == 1:
            local_estimates = local_estimates.squeeze(-1)

        # Local kNN operations
        start = timer()
        if not parallel:
            for m, model in enumerate(self.local_models):
                local_estimates[m], knn_distances[m] = model.predict(X)  # (n_queries,)
        else:  # parallel processing
            local_returns = self.pool.map(predict, zip(self.local_models, [X] * self.n_splits))
            local_estimates, knn_distances = [np.array(l) for l in list(zip(*local_returns))]
        elapsed_time = timer() - start
        if self.verbose:
            print("\n\tLocal kNN operations: {:.2f}s / {:.4f}s (per split)".format(
                elapsed_time, elapsed_time / self.n_splits)
            )
        return local_estimates, knn_distances

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_random_split(data, n_splits):
        # assume data is list
        if not hasattr(data, '__getitem__'):
            data = [data]
        N = len(data[0])
        rand_perm = np.random.permutation(N)
        data = [x[rand_perm] for x in data]  # shuffling data
        return [np.array_split(x, n_splits) for x in data]


class SplitSelectKNeighborsRegressor(SplitSelectKNeighbors):
    def __init__(self, thresholding=False, **kwargs):
        super().__init__(**kwargs)
        self.thresholding = thresholding  # if True, take thresholding after each local estimates
        self.local_models = []
        self.base_model = ExtendedKNeighborsRegressor

    def predict(self, X, parallel=False):
        # X: np.array; (n_queries, n_features)
        n_queries = X.shape[0]

        # 1) Local operations
        local_estimates, knn_distances = self.local_predict(X, parallel=parallel)  # (n_samples, n_queries)

        # 2) Global aggregation
        start = timer()

        if self.n_select < self.n_splits:
            # Note: knn_distances.shape = (n_splits, n_queries)
            selected_indices = np.argpartition(knn_distances, self.n_select, axis=0)
            selected_indices = selected_indices[:self.n_select, :]  # (n_selected, n_queries); takes O(n_splits)
            if self.verbose:
                print("\tPick L={} out of M={}: {:.4f}s".format(self.n_select, self.n_splits, timer() - start))
            final_estimate = local_estimates[
                selected_indices,
                np.repeat(np.arange(n_queries).reshape(1, n_queries), self.n_select, axis=0)
            ].mean(axis=0)  # (n_queries)
        else:
            final_estimate = local_estimates.mean(axis=0)  # (n_queries,)

        if self.is_classifier:
            if self.multilabel_classification:
                final_estimate = self.onehot_encoder.inverse_transform(final_estimate).reshape((-1,))
            else:
                final_estimate = (final_estimate > .5)

        return final_estimate


# for multiprocessing
def predict(model_data):
    model, data = model_data
    return model.predict(data)
