import numpy as np
import pandas as pd
import string
from pathlib import Path
from scipy.stats import multivariate_normal
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import miniboone_utils as mb


# TODO: implement original_split method that returns the original train-valid-test split
class Dataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.classification = None
        self.onehot_encoder = None

    @property
    def data(self):
        return self.X, self.y

    @staticmethod
    def load_and_preprocess(*args, **kwargs):
        raise NotImplementedError

    def train_test_split(self, test_size=0.4, seed=0):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed)

        mu_train = X_train.mean(axis=0, keepdims=True)
        sigma_train = X_train.std(axis=0, keepdims=True)
        sigma_train[sigma_train == 0] = 1

        X_train = (X_train - mu_train) / sigma_train
        X_test = (X_test - mu_train) / sigma_train

        return X_train, X_test, y_train, y_test


class MixtureOfTwoGaussians:
    def __init__(self, prior=.5, sigma=1., d=5):
        self.prior = prior
        self.params0 = np.zeros(d), np.eye(d)
        self.params1 = np.ones(d), sigma ** 2 * np.eye(d)
        self.d = d

        self.classification = True
        self.name = 'mog'

    def compute_bayes_error(self, n_samples=100000):
        X = self.generate(n_samples)[0]
        eta = self.eta(X)
        return np.minimum(eta, 1 - eta).mean()

    def eta(self, x):
        # return P(Y=1|X=x)
        if len(x.shape) == 1:  # for 1-dim case
            x = x[:, np.newaxis]
        p0 = multivariate_normal.pdf(x, *self.params0)
        p1 = multivariate_normal.pdf(x, *self.params1)
        return self.prior * p1 / (self.prior * p1 + (1 - self.prior) * p0)

    def generate(self, n_samples):
        # stratified sampling from mixture of Gaussians
        n_samples1 = np.ceil(n_samples * self.prior).astype(int)
        n_samples0 = n_samples - n_samples1
        X0 = np.random.multivariate_normal(*self.params0, size=n_samples0)
        X1 = np.random.multivariate_normal(*self.params1, size=n_samples1)
        y0 = np.zeros(n_samples0)
        y1 = np.ones(n_samples1)
        X = np.concatenate([X0, X1], axis=0)
        y = np.concatenate([y0, y1], axis=0)
        return X, y

    def train_test_split(self, n_train=100000, n_test=10000):
        X_train, y_train = self.generate(n_train)
        X_test, y_test = self.generate(n_test)
        return X_train, y_train, X_test, y_test


class HTRU2(Dataset):
    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'HTRU2'

    @staticmethod
    def load_and_preprocess(root):
        # Reference: https://github.com/AshishSardana/pulsars-candidate-classifier/blob/master/Pulsar%20Classification.ipynb
        filepath = f"{root}/data/HTRU2/HTRU_2.csv"
        df = pd.read_csv(filepath, header=None)
        features = list(df.columns)
        features.remove(8)
        X, y = np.array(df[features]), np.array(df[8])
        return X, y


class CREDIT(Dataset):
    """
    References
    ----------
    [1] https://github.com/meauxt/credit-card-default/blob/master/credit_card_default.ipynb
    [2] https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook
    """
    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'CREDIT'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        xls = pd.ExcelFile(f"{root}/data/CREDIT/credit_cards_dataset.xls")
        df = xls.parse('Data')
        df.columns = df.iloc[0]  # set the first row as column names
        df = df.iloc[1:]  # drop the duplicated first row
        del df['ID']  # drop the ID column
        if verbose:
            print(df)
        X, y = np.array(df[df.columns[:-1]]).astype(float), np.array(df[df.columns[-1]]).astype(int)
        return X, y


class MiniBooNE(Dataset):
    def __init__(self, root='.'):
        super().__init__()
        self.df = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'MiniBooNE'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        filepath = f"{root}/data/miniboone/MiniBooNE_PID.txt"
        df = mb.import_data(Path(filepath))

        # Rename columns
        columns_names = mb.get_column_names(df)
        df.rename(columns=lambda x: columns_names[x], inplace=True)

        # Get the number of electrons and muons in the file
        num_electron, num_muon = mb.get_num_neutrinos(Path(filepath))

        # Add the "target" column (1 - electron, 0 - muon)
        df = mb.add_target_column(df, num_electron, num_muon)

        if verbose:
            # Data inspection
            df.head()
            df.info()
            mb.pretty_describe(df)

        return df

    def train_test_split(self, test_size=0.4, seed=0):
        train, test = mb.train_test_stratified_split(df=self.df, target='target', test_size=test_size, random_state=seed)

        # Split X and y
        X_train, y_train = mb.x_y_split(df=train, y_col='target')
        X_test, y_test = mb.x_y_split(df=test, y_col='target')

        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        mu_train = X_train.mean(axis=0, keepdims=True)
        sigma_train = X_train.std(axis=0, keepdims=True)
        sigma_train[sigma_train == 0] = 1

        X_train = (X_train - mu_train) / sigma_train
        X_test = (X_test - mu_train) / sigma_train

        return X_train, X_test, y_train, y_test


class NewsGroups20(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.X, self.y = fetch_20newsgroups_vectorized(subset='all', return_X_y=True)
        self.X = self.X.toarray()
        self.onehot_encoder = OneHotEncoder().fit(np.arange(20).reshape((-1, 1)))
        self.classification = True
        self.name = 'NewsGroups20'


class GISETTE(Dataset):
    # Reference
    # [1] https://github.com/aashsach/random-forest/

    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'GISETTE'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        with open(f"{root}/data/GISETTE/gisette_train.data") as f:
            data = [row.strip().split(" ") for row in f.readlines()]
        X_train = np.array(data).astype(int)

        with open(f"{root}/data/GISETTE/gisette_train.labels") as f:
            classes = [row.strip().split(" ") for row in f.readlines()]
        y_train = (np.array(classes).astype(int).squeeze() == np.ones(len(classes))).astype(int)

        with open(f"{root}/data/GISETTE/gisette_valid.data") as f:
            data = [row.strip().split(" ") for row in f.readlines()]
        X_valid = np.array(data).astype(int)

        with open(f"{root}/data/GISETTE/gisette_valid.labels") as f:
            classes = [row.strip().split(" ") for row in f.readlines()]
        y_valid = np.array(classes).astype(int)
        y_valid = y_valid[:, 0]
        y_valid[y_valid == -1] = 0  # somehow negative label in valid set is -1

        X = np.concatenate([X_train, X_valid], axis=0)
        y = np.concatenate([y_train, y_valid], axis=0)

        return X, y


class SUSY(Dataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/SUSY
    """

    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'SUSY'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        filename = f'{root}/data/SUSY/SUSY.csv.gz'
        df = pd.read_csv(filename, compression='gzip', header=None, sep=',', quotechar='"', error_bad_lines=False)
        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])

        return X, y


class HIGGS(Dataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/HIGGS
    """

    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'HIGGS'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        filename = f'{root}/data/HIGGS/HIGGS.csv.gz'
        df = pd.read_csv(filename, compression='gzip', header=None, sep=',', quotechar='"', error_bad_lines=False)
        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])

        return X, y


class BNGLetter(Dataset):
    """
    References
    ----------
    [1] https://www.openml.org/d/1378
    """
    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'BNGLetter'
        self.onehot_encoder = OneHotEncoder().fit(np.array(list(string.ascii_uppercase)).reshape((-1, 1)))

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        filename = f'{root}/data/BNGLetter/BNG_letter_1000_1.csv'
        df = pd.read_csv(filename, header=0, sep=',', quotechar='"', error_bad_lines=False)
        X, y = np.array(df[df.columns[:-1]]), np.array(df[df.columns[-1]])

        return X, y


class WineQuality(Dataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/wine+quality
    """
    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = False
        self.name = 'WineQuality'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        red = pd.read_csv(f'{root}/data/WineQuality/winequality-red.csv', low_memory=False, sep=';')
        white = pd.read_csv(f'{root}/data/WineQuality/winequality-white.csv', low_memory=False, sep=';')
        df = pd.concat([red, white], ignore_index=True)
        df = white
        X, y = np.array(df[df.columns[:-1]]), np.array(df[df.columns[-1]])

        return X, y


class YearPredictionMSD(Dataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    """
    def __init__(self, root='.'):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = False
        self.name = 'YearPredictionMSD'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        filename = f'{root}/data/YearPredictionMSD/YearPredictionMSD.txt.zip'
        df = pd.read_csv(filename, compression='zip', header=None, sep=',', quotechar='"', error_bad_lines=False)
        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])

        return X, y
