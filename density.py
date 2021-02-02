from timeit import default_timer as timer

import numpy as np
from scipy.special import gamma, logsumexp
from sklearn.neighbors._base import _check_weights, NeighborsBase, KNeighborsMixin, UnsupervisedMixin
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from regressor import SplitKNeighbors


def find_unit_volume(d, p=2):
    # return the volume of the d-dim. unit ball under the L^p norm
    return (2 * gamma(1 + 1 / p)) ** d / gamma(1 + d / p)


class KNeighborsDensity(NeighborsBase, KNeighborsMixin, UnsupervisedMixin):
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
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self.weights = _check_weights(weights)
        assert weights == 'uniform', 'a weighted version needs further investigation'
        assert metric == 'minkowski', 'only volumes of minkowski norm ball can be analytically computed'

    def fit(self, X, y=None):
        self.d = X.shape[1]
        self.unit_vol = find_unit_volume(self.d, self.p)

        return super().fit(X, y)

    def predict(self, X, k=None, k_shift=0):
        return self.log_normalized_volume(X, k, k_shift)

    def log_normalized_volume(self, X, k=None, k_shift=0):
        """Evaluate the log density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).
        k: array_like, shape (n_k,) or None
            An array of k values with which density estimates would be evaluated
            If None, use the maximum neighbor distance.

        Returns
        -------
        log_U : ndarray, shape (n_samples, n_k)
            The array of log(normalized_volume) evaluations.
        k: ndarray, shape (n_k,)
            The array of k values used to compute log_U
        """
        check_is_fitted(self)
        if k is None:
            k = np.array([self.n_neighbors])
        else:
            k = np.array(k)
        assert max(k) + k_shift <= self.n_neighbors

        # Find k-NN distances
        neigh_dist, neigh_ind = self.kneighbors(X)
        neigh_dist = neigh_dist[:, k_shift:]
        neigh_dist = neigh_dist[:, k - 1]

        # Compute normalized volumes
        log_U = np.log(self.n_samples_fit_ - k_shift) \
                + np.log(self.unit_vol) \
                + self.d * np.log(neigh_dist)  # (n_samples, n_k)

        if len(k) == 1:
            log_U = log_U.squeeze(-1)
            neigh_dist = neigh_dist.squeeze(-1)

        return log_U, neigh_dist

    def score_samples(self, X=None, k=None):
        """Evaluate the log density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).
        k: array_like, shape (n_k,) or None
            An array of k values with which density estimates would be evaluated
            If None, use the maximum neighbor distance.

        Returns
        -------
        log_density : ndarray, shape (n_samples, n_k)
            The array of log(density) evaluations.
        """
        if X is None:
            X = self._fit_X
            k_shift = 1  # to exclude trivial zero 1-NN distances
        else:
            X = check_array(X, accept_sparse='csr')
            k_shift = 0

        if k is None:
            k = np.array([self.n_neighbors])
        else:
            k = np.array(k)

        log_U, neigh_dist = self.log_normalized_volume(X, k, k_shift=k_shift)
        return np.log(k - 1) - log_U

    def score(self, X=None, y=None, k=None):
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
        log_prob : ndarray, shape (n_k,)
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.score_samples(X, k), axis=-1)

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


class SplitKNeighborDensity(SplitKNeighbors):
    def __init__(self, **kwargs):
        # algorithm: one of {'auto', 'ball_tree', 'kd_tree', 'brute'}
        super().__init__(**kwargs)
        self.local_models = []
        self.base_model = KNeighborsDensity

    def score_samples(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, X, k=None, parallel=False):
        # X: np.array; (n_queries, n_features)
        n_queries = X.shape[0]
        if k is None:
            k = np.array([self.n_neighbors])
        else:
            k = np.array(k)
            assert max(k) <= self.n_neighbors

        # Local operations
        local_log_volumes, knn_distances = self.local_predict(X, k, parallel)  # (n_samples, n_queries, n_k)

        # Global aggregation
        # Note that knn_distances.shape = (n_samples, n_queries, n_k)
        start = timer()
        selected_indices = np.argpartition(knn_distances, self.n_select, axis=0)[:self.n_select, :]  # (n_selected, n_queries); takes O(n_splits)
        print("\tFind L distances: {:.4f}s".format(timer() - start))

        selected_log_volumes = local_log_volumes[
            selected_indices,
            np.repeat(np.arange(n_queries).reshape(1, n_queries), self.n_select, axis=0)
        ]  # (self.n_select, n_queries, n_k)

        final_estimates = dict()  # each entry has shape (n_k, n_queries)
        final_estimates['agg_selective1'] = np.log(k * self.n_select - 1) - logsumexp(selected_log_volumes, axis=0)
        final_estimates['mean_selective1'] = ((np.log(k - 1) - selected_log_volumes).mean(0))
        final_estimates['agg_selective0'] = np.log(k * self.n_splits - 1) - logsumexp(local_log_volumes, axis=0)
        final_estimates['mean_selective0'] = ((np.log(k - 1) - local_log_volumes).mean(0))

        return final_estimates


# for multiprocessing
def predict(model_data_k):
    model, data_k = model_data_k
    X, k = data_k
    return model.log_normalized_volume(X, k)
