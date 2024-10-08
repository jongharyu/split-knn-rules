#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import multiprocessing as mp
import pickle
import pprint as pp
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from timeit import default_timer as timer

import cpuinfo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import datasets
from regressor import SplitSelectKNeighborsRegressor
from utils import str2bool, Logger
from validation import GridSearchForKNeighborsEstimator, GridSearchForSplitSelectKNeighborsEstimator
from validation import compute_error

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
parser.add_argument('--n-neighbors', type=int, default=1)
parser.add_argument('--select-ratio', type=float, default=None)
parser.add_argument('--search-select-ratio', action='store_true')
parser.add_argument('--no-standard', action='store_false')
parser.add_argument('--parallel', type=str2bool, default=False, metavar='P',
                    help='use multiprocessors')
parser.add_argument('--dataset', type=str, default='MiniBooNE',
                    choices=['MiniBooNE',
                             'HTRU2',
                             'CREDIT',
                             'GISETTE',
                             'SUSY',
                             'HIGGS',
                             'NewsGroups20',
                             'BNGLetter',
                             'WineQuality',
                             'GasTurbine',
                             'YearPredictionMSD'])
parser.add_argument('--main-path', type=str, default='.',
                    help='main path where datasets live and loggings are saved')
parser.add_argument('--k-max', type=int, default=1024)
parser.add_argument('--fine-search', type=str2bool, default=False)
parser.add_argument('--n-folds', type=int, default=5)
parser.add_argument('--temp', action='store_true')
parser.add_argument('--verbose', type=bool, default=True)

if __name__ == '__main__':
    mp.set_start_method("spawn")

    args = parser.parse_args()
    if args.parallel:
        print("Parallel processing...")

    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    experiment_dir = Path(f'{args.main_path}/results/{args.dataset}/{timestamp}')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_path = str(experiment_dir)
    if args.temp:
        run_path = mkdtemp(dir=run_path)
    sys.stdout = Logger(f'{run_path}/run.log')

    # load datasets
    print("Loading data... ", end='')
    start = timer()
    dataset = getattr(datasets, args.dataset)(root=args.main_path)
    print(f"done ({timer() - start:.2f}s)")

    print(f'Path: {run_path}')
    print(f'Time: {timestamp}')
    print(f'Args: {args}')
    info = cpuinfo.get_cpu_info()
    del info['flags']
    print('CPUInfo: ')
    pp.pprint(info)

    n_trials = args.n_trials
    keys = ['standard_1NN',
            'standard_kNN',
            'split_select_1NN',
            'split_1NN']
    error_rates = defaultdict(partial(np.zeros, n_trials))
    train_times = defaultdict(partial(np.zeros, n_trials))
    test_times = defaultdict(partial(np.zeros, n_trials))
    model_selection_times = defaultdict(partial(np.zeros, n_trials))
    best_params = defaultdict(partial(np.zeros, n_trials))

    validation_profiles = dict(
        standard_kNN={n: None for n in range(n_trials)},
        split_select_1NN={n: None for n in range(n_trials)},
        split_1NN={n: None for n in range(n_trials)},
    )

    for n in range(n_trials):
        # Split dataset at random
        X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=args.test_size, seed=n)

        # Truncate test set to limit time complexity of experiments
        X_test, y_test = X_test[:args.max_test_size], y_test[:args.max_test_size]
        if n == 0:
            print(f"Data size: train={X_train.shape[0]}, test={X_test.shape[0]}")

        # set maximum k
        k_max = np.min([args.k_max, X_train.shape[0] / 25])

        print(f"Trial #{n + 1}/{n_trials}")
        # Standard k-NN rules
        if args.no_standard:
            for key in ['standard_1NN', 'standard_kNN']:
                k_opt = 1
                model_selection_time = 0.
                if key == 'standard_kNN':
                    start = timer()
                    k_opt, validation_profiles[key][n] = \
                        GridSearchForKNeighborsEstimator(
                            n_folds=args.n_folds,
                            n_repeat=1,
                            max_valid_size=args.max_test_size,
                            verbose=args.verbose,
                            classification=dataset.classification,
                        ).grid_search(
                            X_train,
                            y_train,
                            fine_search=args.fine_search,
                            k_max=k_max)
                    model_selection_time = timer() - start
                    print(f'....{args.n_folds}-fold CV ({model_selection_time:.2f}s)')
                best_params[key] = k_opt
                model_selection_times[key][n] = model_selection_time
                start = timer()
                Predictor = KNeighborsClassifier if dataset.classification else KNeighborsRegressor
                predictor = Predictor(n_neighbors=k_opt,
                                      n_jobs=-1 if args.parallel else None,
                                      algorithm=args.algorithm)
                predictor.fit(X_train, y_train)
                train_times[key][n] = timer() - start
                start = timer()
                print(f'\t{key} (k={k_opt}; {predictor._fit_method}): ', end='')
                y_test_pred = predictor.predict(X_test)
                test_times[key][n] = timer() - start
                error_rates[key][n] = compute_error(y_test_pred, y_test, dataset.classification)

                print(f"{error_rates[key][n]:.4f} (tr: {train_times[key][n]:.2f}s, te: {test_times[key][n]:.2f}s)")

        # Split rules
        split_keys = ['split_select_1NN', 'split_1NN']
        select_ratios = [args.select_ratio, 1.0]
        with mp.get_context("spawn").Pool() as pool:
            for (split_key, select_ratio) in zip(split_keys, select_ratios):
                validation_profiles[split_key][n] = dict(n_splits=None, select_ratio=None)
                n_splits_opt, validation_profiles[split_key][n]['n_splits'], \
                select_ratio_opt, validation_profiles[split_key][n]['select_ratio'] \
                    = GridSearchForSplitSelectKNeighborsEstimator(
                    n_folds=args.n_folds,
                    n_repeat=1,
                    max_valid_size=args.max_test_size,
                    parallel=args.parallel,
                    verbose=args.verbose,
                    classification=dataset.classification,
                    onehot_encoder=dataset.onehot_encoder,
                    n_neighbors=args.n_neighbors,
                    pool=pool,
                ).grid_search(
                    X_train,
                    y_train,
                    n_splits_max=k_max,
                    fine_search=args.fine_search,
                    select_ratio=select_ratio,
                    search_select_ratio=True if (args.search_select_ratio and select_ratio < 1) else False,
                )
                model_selection_time = timer() - start
                print(f'....{args.n_folds}-fold CV ({model_selection_time:.2f}s)')

                start = timer()
                estimator = SplitSelectKNeighborsRegressor(
                    n_neighbors=1,
                    n_splits=n_splits_opt,
                    select_ratio=select_ratio_opt,
                    n_select=None,
                    algorithm=args.algorithm,
                    verbose=False,
                    classification=dataset.classification,
                    onehot_encoder=dataset.onehot_encoder,
                    pool=pool,
                ).fit(X_train, y_train)
                train_times[split_key][n] = timer() - start

                start = timer()
                print(f'\t{split_key} (M={n_splits_opt}, kappa={select_ratio_opt if select_ratio_opt else -1:.2f}; {estimator._fit_method}): ', end='')
                y_test_pred = estimator.predict(X_test, parallel=args.parallel)
                test_times[split_key][n] = timer() - start

                model_selection_times[split_key][n] = model_selection_time
                best_params[split_key][n] = n_splits_opt
                error_rates[split_key][n] = compute_error(y_test_pred, y_test, dataset.classification)
                print(f"{error_rates[split_key][n]:.4f} (tr: {train_times[split_key][n]:.2f}s, te: {test_times[split_key][n]:.2f}s)")

    # Store data (serialize)
    data = dict(keys=keys,
                train_times=train_times,
                test_times=test_times,
                error_rates=error_rates,
                model_selection_times=model_selection_times,
                validation_profiles=validation_profiles,
                cpu_info=cpuinfo.get_cpu_info(),
                args=args)
    filename = f'{run_path}/{dataset.name}_test{args.test_size}_{args.n_trials}tr_{mp.cpu_count()}cores_alg{args.algorithm}.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot validation profiles
    colors = ['red', 'blue', 'green']
    for i, key in enumerate(['standard_kNN', 'split_select_1NN', 'split_1NN']):
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
    plt.title(f'{args.dataset} ({n_trials} runs)')
    plt.legend()
    plt.savefig(f'{run_path}/validation_profile.pdf')
    plt.close()

    if validation_profiles['split_select_1NN'][0]['select_ratio'] is not None:
        param_set = validation_profiles['split_select_1NN'][0]['select_ratio'][0]
        errs = np.zeros((n_trials, len(param_set)))
        for n in range(n_trials):
            errs[n] = validation_profiles['split_select_1NN'][n]['select_ratio'][1]
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
        plt.title(f'{args.dataset} ({n_trials} runs)')
        plt.legend()
        plt.savefig(f'{run_path}/validation_profile_select_ratio.pdf')
