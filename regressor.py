import multiprocessing as mp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors._base import _check_weights, NeighborsBase, KNeighborsMixin, UnsupervisedMixin
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from timeit import default_timer as timer

def compute_error_rate(truth, prediction):
    wrong = 0
    for x, y in zip(truth, prediction):
        if x != y:
            wrong = wrong + 1
    return wrong / len(truth)


def get_random_split(num_splits, data):
    # assume data is iterable
    N = len(data[0])
    rand_perm = np.random.permutation(N)
    data = [x[rand_perm] for x in data]  # shuffling data
    return [np.array_split(x, num_splits) for x in data]


from sklearn.neighbors._base import _get_weights
from sklearn.utils import check_array
class ModifiedKNeighborsRegressor(KNeighborsRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

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


class SplitKNeighborsRegressor:
    def __init__(self, n_neighbors, kappa=0.9, distance_selective=True, thresholding=False, algorithm='auto'):
        # algorithm: one of {'auto', 'ball_tree', 'kd_tree', 'brute'}

        self.distance_selective = distance_selective  # if True, select top local estimates based on knn distances
        self.thresholding = thresholding  # if True, take thresholding after each local estimates

        self.n_neighbors = n_neighbors
        self.kappa = kappa
        self.algorithm = algorithm

        self.sigma = 1 - np.exp(-n_neighbors / 4)

        self.local_regressors = []
        self._fit_method = None

    def fit(self, split_data):
        # split_data: [splits of X, splits of y]
        self.M = len(split_data[0])
        # TODO: check if classifier.predict_proba is equivalent to regressor.predict for default cases
        self.local_regressors = [
            ModifiedKNeighborsRegressor(n_neighbors=self.n_neighbors, algorithm=self.algorithm).fit(
                split_data[0][i], split_data[1][i],
            )
            for i in range(self.M)
        ]
        self._fit_method = self.local_regressors[0]._fit_method

        if self.distance_selective:
            L = int(np.floor(self.kappa * self.M * self.sigma))  # number of estimates to be selected
            self.L = max([L, 1])
        else:
            self.L = self.M

        return self

    def predict(self, X, parallel=False):
        # X: np.array; (num_queries, num_features)
        num_queries = X.shape[0]
        local_estimates = np.zeros((self.M, num_queries))  # kNN regression estimates
        knn_distances = np.zeros((self.M, num_queries))  # kNN distances to the query

        start = timer()
        if not parallel:
            # Local kNN operations
            for m, regressor in enumerate(self.local_regressors):
                local_estimates[m], knn_distances[m] = regressor.predict(X)  # (num_queries,)
        else:  # parallel processing
            with mp.Pool() as pool:
                local_returns = pool.map(predict, zip(self.local_regressors, [X] * self.M))
            local_estimates, knn_distances = [np.array(l) for l in list(zip(*local_returns))]

        elapsed_time = timer() - start
        print("\n\tLocal kNN operations: {:.2f}s / {:.4f}s (per split)".format(elapsed_time, elapsed_time / self.M))

        if self.thresholding:
            local_estimates = (local_estimates > 0.5)

        # Global aggregation
        if self.distance_selective:
            start = timer()
            chosen_indices = np.argpartition(knn_distances, self.L, axis=0)[:self.L, :]  # (L, num_queries); takes O(M)
            elapsed_time = timer() - start
            print("\tFind L distances: {:.4f}s".format(elapsed_time))

            # start = timer()
            final_estimate = local_estimates[
                chosen_indices,
                np.repeat(np.arange(num_queries).reshape(1, num_queries), self.L, axis=0)
            ].mean(axis=0)  # (num_queries)
            # elapsed_time = timer() - start
            # print("Select and compute mean: {:.4f}s".format(elapsed_time))
        else:
            final_estimate = local_estimates.mean(axis=0)  # (num_queries,)

        return final_estimate


# for multiprocessing
def predict(model_data):
    model, data = model_data
    return model.predict(data)


class KNeighborDensity(NeighborsBase, KNeighborsMixin,
                       UnsupervisedMixin):
    """Density estimation based on k-nearest neighbors.

    Read more in the :ref:`User Guide <density>`.

    .. versionadded:: ???

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : str or callable, default='minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric to use. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> neigh = KNeighborsRegressor(n_neighbors=2)
    >>> neigh.fit(X, y)
    KNeighborsRegressor(...)
    >>> print(neigh.predict([[1.5]]))
    [0.5]

    See also
    --------
    NearestNeighbors
    RadiusNeighborsRegressor
    KNeighborsClassifier
    RadiusNeighborsClassifier

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    .. warning::

       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances but
       different labels, the results will depend on the ordering of the
       training data.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """
    @_deprecate_positional_args
    def __init__(self, n_neighbors=5, *, weights='uniform',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        from sklearn.neighbors import KernelDensity
        super().__init__(
              n_neighbors=n_neighbors,
              algorithm=algorithm,
              leaf_size=leaf_size, metric=metric, p=p,
              metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self.weights = _check_weights(weights)

    def score_samples(self, X):
        """Evaluate the log density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        check_is_fitted(self)
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

        # return y_pred

        X = check_array(X, order='C', dtype=DTYPE)
        if self.tree_.sample_weight is None:
            N = self.tree_.data.shape[0]
        else:
            N = self.tree_.sum_weight
        atol_N = self.atol * N
        log_density = self.tree_.kernel_density(
            X, h=self.bandwidth, kernel=self.kernel, atol=atol_N,
            rtol=self.rtol, breadth_first=self.breadth_first, return_log=True)
        log_density -= np.log(N)
        return log_density

    def score(self, X, y=None):
        """Compute the total log probability density under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.score_samples(X))


    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is not implemented. See KernelDensity.sample.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term: `Glossary <random_state>`.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        """
        raise NotImplementedError
