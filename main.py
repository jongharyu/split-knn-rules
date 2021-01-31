#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import numpy as np
from multiprocessing import cpu_count
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from timeit import default_timer as timer

import cpuinfo

import datasets
from regressor import SplitSelectKNeighborsRegressor
from validation import compute_error_rate
from utils import str2bool
from validation import GridSearchWithCrossValidationForKNeighborsClassifier

# mpl.style.use( 'ggplot' )
markers = ['o', 's', '*', 'v', '^', 'D', 'h', 'x', '+', '8', 'p', '<', '>', 'd', 'H', 1, 2, 3, 4]

parser = argparse.ArgumentParser(description='Split knn rules')
parser.add_argument('--test-size', type=float, default=0.4, metavar='t',
                    help='test split ratio')
parser.add_argument('--n-trials', type=int, default=1,
                    help='number of different train/test splits')
parser.add_argument('--algorithm', type=str, default='auto',
                    help='knn search algorithm (default: "auto")')
parser.add_argument('--parallel', type=str2bool, default=False, metavar='P',
                    help='use multiprocessors')
parser.add_argument('--dataset', type=str, default='MiniBooNE',
                    choices=['MiniBooNE', 'HTRU2', 'CREDIT', 'GISETTE',
                             'SUSY', 'HIGGS', 'BNGLetter',
                             'WineQuality', 'YearPredictionMSD'])
parser.add_argument('--main-path', type=str, default='.')
parser.add_argument('--k-standard-max', type=int, default=1025)

args = parser.parse_args()
dataset = getattr(datasets, args.dataset)(root=args.main_path)

def run():
    if args.parallel:
        print("Parallel processing...")

    n_trials = args.n_trials
    # ks = [1] + [2 ** logk + 1 for logk in range(1, np.ceil(np.log2(args.k_standard_max)).astype(int))]
    # base_k = [1, 3]  # for split methods
    keys = ['standard_1NN', 'standard_kNN_CV', 'Msplit_1NN_CV']
    error_rates = {key: np.zeros(n_trials) for key in keys}
    elapsed_times = {key: np.zeros(n_trials) for key in keys}

    for n in range(n_trials):
        # Split dataset at random
        X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=args.test_size, seed=n)

        # Standard k-NN rules
        for key in ['standard_1NN', 'standard_kNN_CV']:
            n_neighbors = 1
            if key == 'standard_kNN_CV':
                start = timer()
                n_neighbors = GridSearchWithCrossValidationForKNeighborsClassifier(n_folds=5, n_repeat=1).grid_search(X_train, y_train)
                print("Grid search with 5-fold CV: k={} (Elpased time = {:.2f}s)".format(n_neighbors, timer() - start))

            start = timer()
            Predictor = KNeighborsClassifier if dataset.classification else KNeighborsRegressor
            predictor = Predictor(n_neighbors=n_neighbors,
                                  n_jobs=-1 if args.parallel else None,
                                  algorithm=args.algorithm)
            predictor.fit(X_train, y_train)
            print('\t{} (k={}): '.format(key, n_neighbors), end='')
            y_test_pred = predictor.predict(X_test)
            elapsed_times[key][n] = timer() - start
            error_rates[key][n] = compute_error_rate(y_test_pred, y_test)

            print("\t{:.4f} ({:.2f}s)".format(error_rates[key][n], elapsed_times[key][n]))


        # Split rules
        key = 'Msplit_1NN_CV'
        start = timer()
        n_splits = np.min([5 * n_neighbors, np.sqrt(X_train.shape[0])]).astype(int)
        # print("Grid search with 5-fold CV: M={}, kappa={} (Elpased time = {:.2f}s)".format(n_splits, select_ratio, timer() - start))
        regressor = SplitSelectKNeighborsRegressor(n_neighbors=1,
                                                   n_splits=n_splits,
                                                   select_ratio=None,
                                                   algorithm=args.algorithm
                                                   ).fit(X_train, y_train)
        print('\t{} (M={}): '.format(key, n_splits), end='')
        y_test_pred = regressor.predict(X_test, parallel=args.parallel)
        elapsed_time = timer() - start
        for key_ in y_test_pred:
            if key_.startswith('Msplit_select1'):
                y_test_pred[key_] = (y_test_pred[key_] > .5) if dataset.classification else y_test_pred[key_]
                error_rates[key][n] = compute_error_rate(y_test_pred[key_], y_test)
                elapsed_times[key][n] = elapsed_time
        print("\t\t{:.4f} ({:.2f}s)".format(error_rates[key][n], elapsed_times[key][n]))

    # Store data (serialize)
    data = dict(keys=keys,
                elapsed_times=elapsed_times, error_rates=error_rates,
                cpu_info=cpuinfo.get_cpu_info())
    filename = 'results/{}_test{}_{}tr_{}cores_alg{}.pickle'.format(
        dataset.name, args.test_size, args.n_trials, cpu_count(),
        args.algorithm,
    )
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    run()
