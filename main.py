import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from timeit import default_timer as timer

import datasets
from modules import compute_error_rate, get_random_split, SplitKNeighborsRegressor
from utils import str2bool


mpl.style.use( 'ggplot' )
markers = ['o', 's', '*', 'v', '^', 'D', 'h', 'x', '+', '8', 'p', '<', '>', 'd', 'H', 1, 2, 3, 4]

parser = argparse.ArgumentParser(description='Split knn rules')
parser.add_argument('--test-size', type=float, default=0.4, metavar='t',
                    help='test split ratio')
parser.add_argument('--n-trials', type=str, default=10,
                    help='number of different train/test splits')
parser.add_argument('--parallel', type=str2bool, default=False, metavar='P',
                    help='use multiprocessors')
parser.add_argument('--dataset', type=str, default='MiniBooNE',
                    choices=['MiniBooNE', 'HTRU2'])
parser.add_argument('--main-path', type=str, default='.')

args = parser.parse_args()



dataset = getattr(datasets, args.dataset)(root=args.main_path)

def parse_descriptor(key):
    """
                 | distance_selective
    thresholding | 0              1
    -------------------------------------------
               0 | soft_big_NN    split_NN
               1 | big_NN         hard_split_NN
    """

    if 'split' in key:
        distance_selective = True
        thresholding = True if 'hard' in key else False
    else:
        distance_selective = False
        thresholding = False if 'soft' in key else True

    n_neighbors = int(key.split('_')[-1][0])

    return distance_selective, thresholding, n_neighbors


def run():
    if args.parallel:
        print("Parallel processing...")

    n_trials = args.n_trials

    keys = ['split_1NN', 'soft_big_1NN', 'big_1NN',
            'split_3NN', 'hard_split_3NN', 'soft_big_3NN', 'big_3NN',
            'oracle_kNN',]
    ks = [1, 3, 5, 9, 17, 33, 65, 129, 257]
    error_rates = {key: np.zeros((len(ks), n_trials)) for key in keys}
    elapsed_times = {key: np.zeros((len(ks), n_trials)) for key in keys}

    for key in keys:
        for i, k_oracle in enumerate(ks):
            print('Running {} with {}'.format(key, k_oracle), end='; ')
            for n in range(n_trials):
                print(n, end=' ')
                # Split dataset at random
                X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=args.test_size, seed=n)

                start = timer()
                if key.startswith('oracle'):
                    Predictor = KNeighborsClassifier if dataset.classification else KNeighborsRegressor
                    predictor = Predictor(n_neighbors=k_oracle)
                    predictor.fit(X_train, y_train)
                    y_test_pred = predictor.predict(X_test)

                else:
                    if k_oracle == 1:
                        break
                    distance_selective, thresholding, n_neighbors = parse_descriptor(key)
                    regressor = SplitKNeighborsRegressor(n_neighbors=n_neighbors,
                                                         distance_selective=distance_selective,
                                                         thresholding=thresholding)
                    n_splits = k_oracle
                    P = get_random_split(n_splits, [X_train, y_train])
                    regressor.fit(P)
                    y_test_pred = regressor.predict(X_test, parallel=args.parallel)
                    if dataset.classification:
                        y_test_pred = y_test_pred > 1 / 2

                elapsed_times[key][i, n] = timer() - start
                error_rates[key][i, n] = compute_error_rate(y_test_pred, y_test)

            print(elapsed_times[key][i, :].mean())
        print(error_rates[key])

    # Store data (serialize)
    import pickle
    data = dict(ks=ks, keys=keys, elapsed_times=elapsed_times, error_rates=error_rates)
    filename = '{}_test{}.pickle'.format(dataset.name, args.test_size)
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # for i, ks in enumerate()
    for i, key in enumerate(keys):
        plt.errorbar(ks if key not in ['split_1NN'] else np.array(ks),
                     error_rates[key].mean(axis=1),
                     error_rates[key].std(axis=1),
                     marker=markers[i], capsize=3, label=key)
        plt.axhline(error_rates['oracle_1NN'].mean())
        plt.axhline(error_rates['soft_big_1NN'].mean(axis=1).min())
        plt.axhline(error_rates['oracle_kNN'].mean(axis=1).min())
    plt.legend()
    plt.xscale('log', nonposx='clip')
    plt.grid('on')
    plt.title('{} ({} different {}/{} splits)'.format(args.dataset,
                                                      args.n_trials,
                                                      int(100 * (1 - args.test_size)),
                                                      int(100 * args.test_size)))


if __name__ == '__main__':
    run()