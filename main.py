#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import matplotlib as mpl
import numpy as np
from multiprocessing import cpu_count
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from timeit import default_timer as timer

import cpuinfo

import datasets
from regressor import compute_error_rate, SplitKNeighborsRegressor
from utils import generate_keys, str2bool


mpl.style.use( 'ggplot' )
markers = ['o', 's', '*', 'v', '^', 'D', 'h', 'x', '+', '8', 'p', '<', '>', 'd', 'H', 1, 2, 3, 4]

parser = argparse.ArgumentParser(description='Split knn rules')
parser.add_argument('--test-size', type=float, default=0.4, metavar='t',
                    help='test split ratio')
parser.add_argument('--n-trials', type=int, default=10,
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
parser.add_argument('--k-oracle-max', type=int, default=1025)

args = parser.parse_args()
dataset = getattr(datasets, args.dataset)(root=args.main_path)

def run():
    if args.parallel:
        print("Parallel processing...")

    n_trials = args.n_trials
    ks = [1] + [2 ** logk + 1 for logk in range(1, np.ceil(np.log2(args.k_oracle_max)).astype(int))]
    base_k = [1, 3, 11, 31]  # for split methods
    keys = ['oracle_kNN'] + generate_keys(base_k)
    error_rates = {key: np.zeros((len(ks), n_trials)) for key in keys}
    elapsed_times = {key: np.zeros((len(ks), n_trials)) for key in keys}

    # Oracle kNN
    key = 'oracle_kNN'
    for n in range(n_trials):
        # Split dataset at random
        X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=args.test_size, seed=n)
        for i, k_oracle in enumerate(ks):
            start = timer()
            Predictor = KNeighborsClassifier if dataset.classification else KNeighborsRegressor
            predictor = Predictor(n_neighbors=k_oracle,
                                  n_jobs=-1 if args.parallel else None,
                                  algorithm=args.algorithm)
            predictor.fit(X_train, y_train)
            print('Running oracle with k={} ({}/{}) using {}'.format(
                k_oracle, n + 1, n_trials,
                predictor._fit_method), end=': '
            )
            y_test_pred = predictor.predict(X_test)
            elapsed_times[key][i, n] = timer() - start
            error_rates[key][i, n] = compute_error_rate(y_test_pred, y_test)

            print("\tError rate = {:.4f} ({:.2f}s)".format(error_rates[key][i, n], elapsed_times[key][i, n]))
        print("\tAverage error rates: {:.4f}".format(error_rates[key][i, :].mean()))
    print(key, error_rates[key])

    for n_neighbors in base_k:
        for i, n_splits in enumerate(ks):
            for n in range(n_trials):
                if n_splits == 1:
                    break

                # Split dataset at random
                X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=args.test_size, seed=n)
                regressor = SplitKNeighborsRegressor(n_neighbors=n_neighbors, algorithm=args.algorithm)
                X_split, y_split = regressor.get_random_split([X_train, y_train], n_splits)

                start = timer()
                regressor.fit(X_split, y_split)
                print('Running split-{}NN rules with {} splits ({}/{}) using {}'.format(
                    n_neighbors, n_splits,
                    n + 1, n_trials,
                    regressor._fit_method), end=': ')
                y_test_pred = regressor.predict(X_test, parallel=args.parallel)
                elapsed_time = timer() - start
                for key in y_test_pred:
                    y_test_pred[key] = (y_test_pred[key] > .5) if dataset.classification else y_test_pred[key]
                    elapsed_times[key][i, n] = elapsed_time
                    error_rates[key][i, n] = compute_error_rate(y_test_pred[key], y_test)
                    print("\tError rate ({}) = {:.4f} ({:.2f}s)".format(key, error_rates[key][i, n], elapsed_times[key][i, n]))
            else:
                for key in y_test_pred:
                    print("\tAverage error rates ({}): {:.4f}".format(key, error_rates[key][i, :].mean()))
        else:
            for key in y_test_pred:
                print(key, error_rates[key])

    # Store data (serialize)
    data = dict(ks=ks, keys=keys,
                elapsed_times=elapsed_times, error_rates=error_rates,
                cpu_info=cpuinfo.get_cpu_info())
    filename = '{}_test{}_{}tr_{}cores_alg{}.pickle'.format(
        dataset.name, args.test_size, args.n_trials, cpu_count(),
        args.algorithm,
    )
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    run()
