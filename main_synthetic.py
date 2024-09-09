#!/usr/bin/env python
# coding: utf-8
import pickle
from collections import defaultdict
from functools import partial
from timeit import default_timer as timer

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from datasets import MixtureOfTwoGaussians
from regressor import SplitSelectKNeighborsRegressor
from validation import compute_error

# Experiment 1. Simulation with mixture of Gaussians
k0 = 10
alpha = 1  # Holder smoothness

d = 5
prior = .5  # prior probability for class 0

n_test = 10000
n_trials = 10
n_samples_list = [500, 2500, 12500, 62500, 312500]

base_k = [1, 3]

sigmas = [0.5, 1, 2, 3, 4]

for sigma in sigmas:
    mog = MixtureOfTwoGaussians(prior=prior, sigma=sigma, d=d)

    bayes_error = mog.compute_bayes_error()
    print(f"Bayes error: {bayes_error}")

    keys = [f'standard_{k}NN' for k in base_k] + ['standard_kNN']
    error_rates = defaultdict(partial(np.zeros, (len(n_samples_list), n_trials)))
    elapsed_times = defaultdict(partial(np.zeros, (len(n_samples_list), n_trials)))

    # standard k-NN
    for i, n_samples in enumerate(n_samples_list):
        for n in range(n_trials):
            print(f"N={n_samples} ({n + 1}/{n_trials})")
            X_train, y_train, X_test, y_test = mog.train_test_split(n_samples, n_test)

            # 1) Standard k-NN
            k_standard = k0 * n_samples ** ((2 * alpha) / (2 * alpha + d))  # k for standard NN
            k_standard = np.ceil(k_standard).astype(int)

            for kk, n_neighbors in enumerate(base_k + [k_standard]):
                key = f'standard_{n_neighbors if kk != len(base_k) else "k"}NN'

                start = timer()
                classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
                print(f"\t{key}: ", end="")  # classifier._fit_method
                y_test_pred = classifier.predict(X_test)
                elapsed_times[key][i, n] = timer() - start
                error_rates[key][i, n] = compute_error(y_test_pred, y_test)
                print(f"\t\t\t\t\t{error_rates[key][i, n]:.4f} ({elapsed_times[key][i, n]:.2f}s)")

            # 2) Split rules
            for n_neighbors in base_k:
                split_keys = [f'split_select_{n_neighbors}NN',
                              f'split_{n_neighbors}NN']
                select_ratios = [0.5, 1.0]
                for (split_key, select_ratio) in zip(split_keys, select_ratios):
                    # with mp.get_context("spawn").Pool() as pool:
                    n_splits = k_standard
                    start = timer()
                    regressor = SplitSelectKNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        n_splits=n_splits,
                        select_ratio=select_ratio,
                        n_select=None,
                        algorithm='auto',
                        verbose=False,
                        classification=True,
                        pool=None,
                    ).fit(X_train, y_train)
                    print(f'\t{split_key} (M={n_splits}; kappa={select_ratio}): ', end='')
                    y_test_pred = regressor.predict(X_test, parallel=False)
                    elapsed_time = timer() - start
                    elapsed_times[split_key][i, n] = elapsed_time
                    error_rates[split_key][i, n] = compute_error(y_test_pred, y_test)

                    print(f"\t\t{error_rates[split_key][i, n]:.4f} ({elapsed_times[split_key][i, n]:.2f}s)")
        else:
            print(f"\nN={n_samples}; Average error rates")
            for key in error_rates:
                print(f"\t{key} = {error_rates[key][i, :].mean():.4f} ({elapsed_times[key][i, :].mean():.2f}s)")
            else:
                print('\n')

    for key in keys:
        print(key, error_rates[key].mean(axis=1))

    data = dict(keys=keys, elapsed_times=elapsed_times, error_rates=error_rates, bayes_error=bayes_error)

    # Store data (serialize)
    with open(f'results/mog/mog_d{d}_p{prior}_sigma{sigma}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
