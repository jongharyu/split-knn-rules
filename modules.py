import multiprocessing as mp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
    def __init__(self, n_neighbors, kappa=0.9, distance_selective=True, thresholding=False, algorithm='kd_tree'):
        # algorithms: one of {'auto', 'ball_tree', 'kd_tree', 'brute'}

        self.distance_selective = distance_selective  # if True, select top local estimates based on knn distances
        self.thresholding = thresholding  # if True, take thresholding after each local estimates

        self.n_neighbors = n_neighbors
        self.kappa = kappa
        self.algorithm = algorithm

        self.sigma = 1 - np.exp(-n_neighbors / 4)

        self.local_regressors = []

    def fit(self, split_data):
        # split_data: [splits of X, splits of y]
        self.M = len(split_data[0])
        # TODO: check if classifier.predict_proba is equivalent to regressor.predict for default cases
        self.local_regressors = [
            KNeighborsRegressor(n_neighbors=self.n_neighbors, algorithm=self.algorithm).fit(
                split_data[0][i], split_data[1][i]
            )
            for i in range(self.M)
        ]
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
                local_estimates[m] = regressor.predict(X)
                knn_distances[m] = regressor.kneighbors(X)[0][:, -1]  # (num_queries,)

        else:  # parallel processing
            print("Parallel processing...")
            with mp.Pool() as pool:
                local_returns = pool.map(predict, zip(self.local_regressors, [X] * self.M))
            local_estimates, knn_distances = [np.array(l) for l in list(zip(*local_returns))]

        elapsed_time = timer() - start
        print("Local kNN operations: {} / {} (per split)".format(elapsed_time, elapsed_time / self.M))

        if self.thresholding:
            local_estimates = (local_estimates > 0.5)

        # Global aggregation
        if self.distance_selective:
            start = timer()
            chosen_indices = np.argpartition(knn_distances, self.L, axis=0)[:self.L, :]  # (L, num_queries); takes O(M)
            elapsed_time = timer() - start
            print("Find L distances: {}".format(elapsed_time))

            start = timer()
            final_estimate = local_estimates[
                chosen_indices,
                np.repeat(np.arange(num_queries).reshape(1, num_queries), self.L, axis=0)
            ].mean(axis=0)  # (num_queries)
            elapsed_time = timer() - start
            print("Select and comput mean: {}".format(elapsed_time))
        else:
            final_estimate = local_estimates.mean(axis=0)  # (num_queries,)

        return final_estimate


# for multiprocessing
def predict(model_data):
    model, data = model_data
    return model.predict(data), model.kneighbors(data)[0][:, -1]