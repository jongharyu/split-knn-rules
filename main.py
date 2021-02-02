#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import numpy as np
import pickle
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tempfile import mkdtemp
from timeit import default_timer as timer

import cpuinfo
import matplotlib.pyplot as plt

import datasets
from regressor import SplitSelectKNeighborsRegressor
from validation import compute_error_rate
from utils import str2bool, Logger
from validation import GridSearchForKNeighborsClassifier, GridSearchForSplitSelect1NN


# mpl.style.use( 'ggplot' )
markers = ['o', 's', '*', 'v', '^', 'D', 'h', 'x', '+', '8', 'p', '<', '>', 'd', 'H', 1, 2, 3, 4]


# Arguments
parser = argparse.ArgumentParser(description='Split knn rules')
parser.add_argument('--test-size', type=float, default=0.05, metavar='t',
                    help='test split ratio')
parser.add_argument('--max-test-size', type=int, default=1000)
parser.add_argument('--n-trials', type=int, default=1,
                    help='number of different train/test splits')
parser.add_argument('--algorithm', type=str, default='auto',
                    help='knn search algorithm (default: "auto")',
                    choices=['auto', 'ball_tree', 'kd_tree', 'brute'])
parser.add_argument('--search-select-ratio', action='store_true')
parser.add_argument('--parallel', type=str2bool, default=False, metavar='P',
                    help='use multiprocessors')
parser.add_argument('--dataset', type=str, default='MiniBooNE',
                    choices=['MiniBooNE',
                             'HTRU2',
                             'CREDIT',
                             'GISETTE',
                             'SUSY',
                             'HIGGS',
                             'BNGLetter',
                             'WineQuality',
                             'YearPredictionMSD'])
parser.add_argument('--main-path', type=str, default='.',
                    help='main path where datasets live and loggings are saved')
parser.add_argument('--k-max', type=int, default=1024)
parser.add_argument('--n-folds', type=int, default=5)
parser.add_argument('--temp', action='store_true')
parser.add_argument('--verbose', type=bool, default=True)

args = parser.parse_args()


run_id = datetime.datetime.now().isoformat()
experiment_dir = Path('{}/results/{}/'.format(args.main_path, args.dataset))
experiment_dir.mkdir(parents=True, exist_ok=True)
run_path = str(experiment_dir)
if args.temp:
    run_path = mkdtemp(prefix=run_id, dir=run_path)
sys.stdout = Logger('{}/run.log'.format(run_path))
print('Expt: {}'.format(run_path))
print('RunID: {}'.format(run_id))


# select datasets
dataset = getattr(datasets, args.dataset)(root=args.main_path)


def run():
    if args.parallel:
        print("Parallel processing...")

    n_trials = args.n_trials
    keys = ['standard_1NN',
            'standard_kNN',
            'split_select1_1NN',
            'split_select0_1NN',
            ]
    error_rates = defaultdict(partial(np.zeros, n_trials))
    elapsed_times = defaultdict(partial(np.zeros, n_trials))
    model_selection_times = defaultdict(partial(np.zeros, n_trials))
    best_params = defaultdict(partial(np.zeros, n_trials))

    validation_profiles = dict(
        standard_kNN={n: None for n in range(n_trials)},
        Msplit_1NN={n: None for n in range(n_trials)},
    )

    for n in range(n_trials):
        # Split dataset at random
        x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=args.test_size, seed=n)

        # Truncate test set to limit time complexity of experiments
        x_test, y_test = x_test[:args.max_test_size], y_test[:args.max_test_size]
        if n == 0:
            print("Data size: train={}, test={}".format(x_train.shape[0], x_test.shape[0]))

        # set maximum k
        k_max = np.min([args.k_max, x_train.shape[0] / 25])

        print("Trial #{}/{}".format(n + 1, n_trials))
        # Standard k-NN rules
        for key in ['standard_1NN', 'standard_kNN']:
            k_opt = 1
            model_selection_time = 0.
            if key == 'standard_kNN':
                start = timer()
                k_opt, validation_profiles[key][n] = \
                    GridSearchForKNeighborsClassifier(
                        n_folds=args.n_folds,
                        n_repeat=1,
                        max_valid_size=args.max_test_size,
                        verbose=args.verbose,
                ).grid_search(x_train, y_train, k_max=k_max)
                model_selection_time = timer() - start
                print('\t\t{}-fold CV ({:.2f}s)'.format(
                    args.n_folds,
                    model_selection_time))
            best_params[key] = k_opt
            model_selection_times[key] = model_selection_time
            start = timer()
            Predictor = KNeighborsClassifier if dataset.classification else KNeighborsRegressor
            predictor = Predictor(n_neighbors=k_opt,
                                  n_jobs=-1 if args.parallel else None,
                                  algorithm=args.algorithm)
            predictor.fit(x_train, y_train)
            print('\t{} (k={}): '.format(key, k_opt), end='')
            y_test_pred = predictor.predict(x_test)
            elapsed_times[key][n] = timer() - start
            error_rates[key][n] = compute_error_rate(y_test_pred, y_test)

            print("{:.4f} ({:.2f}s)".format(error_rates[key][n], elapsed_times[key][n]))

        # Split rules
        validation_profiles['Msplit_1NN'][n] = dict(n_splits=None, select_ratio=None)
        n_splits_opt, validation_profiles['Msplit_1NN'][n]['n_splits'], \
        select_ratio_opt, validation_profiles['Msplit_1NN'][n]['select_ratio'] \
            = GridSearchForSplitSelect1NN(
            n_folds=args.n_folds,
            n_repeat=1,
            max_valid_size=args.max_test_size,
            parallel=args.parallel,
            verbose=args.verbose,
            classification=dataset.classification,
            onehot_encoder=dataset.onehot_encoder,
        ).grid_search(
            x_train,
            y_train,
            k_max=k_max,
            search_select_ratio=True if dataset.onehot_encoder or args.search_select_ratio else False,
        )
        model_selection_time = timer() - start
        print('\t\t{}-fold CV ({:.2f}s)'.format(
            args.n_folds,
            model_selection_time))

        start = timer()
        estimator = SplitSelectKNeighborsRegressor(
            n_neighbors=1,
            n_splits=n_splits_opt,
            select_ratio=select_ratio_opt,
            algorithm=args.algorithm,
            verbose=False,
            classification=dataset.classification,
            onehot_encoder=dataset.onehot_encoder,
            ).fit(x_train, y_train)

        print('\t{} (M={}, kappa={:.2f}): '.format('Msplit_1NN', n_splits_opt, select_ratio_opt), end='')
        y_test_pred = estimator.predict(x_test, parallel=args.parallel)
        elapsed_time = timer() - start
        for key in y_test_pred:
            model_selection_times[key][n] = model_selection_time
            best_params[key][n] = n_splits_opt
            error_rates[key][n] = compute_error_rate(y_test_pred[key], y_test)
            elapsed_times[key][n] = elapsed_time
        print("{:.4f} ({:.2f}s)".format(error_rates[key][n], elapsed_times[key][n]))

    # Store data (serialize)
    data = dict(keys=keys,
                elapsed_times=elapsed_times, error_rates=error_rates,
                model_selection_times=model_selection_times,
                validation_profiles=validation_profiles,
                cpu_info=cpuinfo.get_cpu_info(),
                args=args)
    filename = '{}/{}_test{}_{}tr_{}cores_alg{}.pickle'.format(
        run_path,
        dataset.name, args.test_size, args.n_trials, cpu_count(),
        args.algorithm,
    )
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot validation profiles
    colors = ['red', 'blue']
    for i, key in enumerate(['standard_kNN', 'Msplit_1NN']):
        if key == 'standard_kNN':
            param_set = validation_profiles[key][0][0]
        else:
            param_set = validation_profiles[key][0]['n_splits'][0]
        errs = np.zeros((n_trials, len(param_set)))
        for n in range(n_trials):
            if key == 'standard_kNN':
                errs[n] = validation_profiles[key][n][1]
            else:
                errs[n] = validation_profiles[key][n]['n_splits'][1]
        plt.plot(param_set,
                 errs.mean(axis=0),
                 linewidth=1,
                 label=key,
                 color=colors[i],
                 marker=markers[i])
        plt.fill_between(param_set,
                         (errs.mean(axis=0) - errs.std(axis=0)),
                         (errs.mean(axis=0) + errs.std(axis=0)),
                         linewidth=0.1,
                         alpha=0.3,
                         color=colors[i])
        plt.xscale('log', nonposx='clip')
    plt.title('{} ({} runs)'.format(args.dataset, n_trials))
    plt.legend()
    plt.savefig('{}/validation_profile.pdf'.format(run_path))
    plt.close()

    if validation_profiles['Msplit_1NN'][0]['select_ratio'] is not None:
        param_set = validation_profiles['Msplit_1NN'][0]['select_ratio'][0]
        errs = np.zeros((n_trials, len(param_set)))
        for n in range(n_trials):
            errs[n] = validation_profiles['Msplit_1NN'][n]['select_ratio'][1]
        plt.plot(param_set,
                 errs.mean(axis=0),
                 linewidth=1,
                 label='select_ratio',
                 marker='x',
                 color='blue')
        plt.fill_between(param_set,
                         (errs.mean(axis=0) - errs.std(axis=0)),
                         (errs.mean(axis=0) + errs.std(axis=0)),
                         linewidth=0.1,
                         alpha=0.3,
                         color='blue')
        plt.xscale('log', nonposx='clip')
        plt.title('{} ({} runs)'.format(args.dataset, n_trials))
        plt.legend()
        plt.savefig('{}/validation_profile_select_ratio.pdf'.format(run_path))

if __name__ == '__main__':
    run()
