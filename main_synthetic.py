#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as timer

from datasets import MixtureOfTwoGaussians
from regressor import compute_error_rate, SplitKNeighborsRegressor
from utils import generate_keys

# Experiment 1. Simulation
k0 = 1
alpha = 1

d = 5
prior = .5  # prior probability for class 0
sigma = 1
mog = MixtureOfTwoGaussians(prior=prior, sigma=sigma, d=d)

bayes_error = mog.compute_bayes_error()
print("Bayes error: {}".format(bayes_error))

n_test = 10000
n_trials = 1
n_samples_list = [500, 2500, 12500, 62500]

base_k = [1, 3]
keys = ['oracle_{}NN'.format(k) for k in base_k] + \
       ['oracle_kNN'] + generate_keys(base_k)
error_rates = {key: np.zeros((len(n_samples_list), n_trials)) for key in keys}
elapsed_times = {key: np.zeros((len(n_samples_list), n_trials)) for key in keys}

# Oracle k-NN
for i, n_samples in enumerate(n_samples_list):
    k_oracle = k0 * n_samples ** ((2 * alpha) / (2 * alpha + 1))  # k for oracle NN
    k_oracle = np.ceil(k_oracle).astype(int)
    for kk, n_neighbors in enumerate(base_k + [k_oracle]):
        key = 'oracle_{}NN'.format(n_neighbors if kk != len(base_k) else 'k')
        for n in range(n_trials):
            X_train, y_train, X_test, y_test = mog.train_test_split(n_samples, n_test)

            start = timer()
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
            print('N={}; running oracle with k={} ({}/{}) using {}'.format(
                n_samples, n_neighbors, n + 1, n_trials,
                classifier._fit_method), end=': '
            )
            y_test_pred = classifier.predict(X_test)
            elapsed_times[key][i, n] = timer() - start
            error_rates[key][i, n] = compute_error_rate(y_test_pred, y_test)

            print("\tError rate = {:.4f} ({:.2f}s)".format(error_rates[key][i, n], elapsed_times[key][i, n]))
        print("\tAverage error rates: {:.4f}".format(error_rates[key][i, :].mean()))
        print(key, error_rates[key])

# Split rules
for n_neighbors in base_k:
    for i, n_samples in enumerate(n_samples_list):
        k_oracle = k0 * n_samples ** ((2 * alpha) / (2 * alpha + 1))  # k for oracle NN
        k_oracle = np.ceil(k_oracle).astype(int)
        n_splits = k_oracle
        for n in range(n_trials):
            X_train, y_train, X_test, y_test = mog.train_test_split(n_samples, n_test)
            regressor = SplitKNeighborsRegressor(n_neighbors=n_neighbors, algorithm='auto')
            train_split = regressor.get_random_split([X_train, y_train], n_splits)

            start = timer()
            regressor.fit(*train_split)
            print('N={}; Running split-{}NN rules with {} splits ({}/{}) using {}'.format(
                n_samples, n_neighbors, n_splits,
                n + 1, n_trials,
                regressor._fit_method), end=': '
            )
            y_test_pred = regressor.predict(X_test)
            elapsed_time = timer() - start
            for key in y_test_pred:
                y_test_pred[key] = y_test_pred[key] > .5
                elapsed_times[key][i, n] = elapsed_time
                error_rates[key][i, n] = compute_error_rate(y_test_pred[key], y_test)
                print("\tError rate = {:.4f} ({:.2f}s)".format(error_rates[key][i, n], elapsed_times[key][i, n]))

        for key in y_test_pred:
            print("\tAverage error rates: {:.4f}".format(error_rates[key][i, :].mean()))

for key in keys:
    print(key, error_rates[key].mean(axis=1))

data = dict(keys=keys, elapsed_times=elapsed_times, error_rates=error_rates, bayes_error=bayes_error)

# Store data (serialize)
with open('mog_d{}_sigma{}.pickle'.format(d, sigma), 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
