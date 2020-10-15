import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import data.miniboone as mb


class HTRU2:
    def __init__(self, root='.'):
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'htru2'

    @staticmethod
    def load_and_preprocess(root):
        # Reference: https://github.com/AshishSardana/pulsars-candidate-classifier/blob/master/Pulsar%20Classification.ipynb
        filepath = "{}/data/HTRU2/HTRU_2.csv".format(root)
        df = pd.read_csv(filepath, header=None)
        features = list(df.columns)
        features.remove(8)
        X, y = np.array(df[features]), np.array(df[8])
        return X, y

    def train_test_split(self, test_size=0.4, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)


class MiniBooNE:
    def __init__(self, root='.'):
        self.df = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'miniboone'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        filepath = "{}/data/miniboone/MiniBooNE_PID.txt".format(root)
        df = mb.import_data(Path(filepath))

        # Rename columns
        columns_names = mb.get_column_names(df)
        df.rename(columns=lambda x: columns_names[x], inplace=True)

        # Get the number of elecrons and muons in the file
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

        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

        return X_train, X_test, y_train, y_test


class CREDIT:
    """
    References
    ----------
    [1] https://github.com/meauxt/credit-card-default/blob/master/credit_card_default.ipynb
    [2] https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook
    """
    def __init__(self, root='.'):
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'CREDIT'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        xls = pd.ExcelFile("{}/data/CREDIT/credit_cards_dataset.xls".format(root))
        df = xls.parse('Data')
        df.columns = df.iloc[0]  # set the first row as column names
        df = df.iloc[1:]  # drop the duplicated first row
        del df['ID']  # drop the ID column
        if verbose:
            print(df)
        X, y = np.array(df[df.columns[:-1]]).astype(float), np.array(df[df.columns[-1]]).astype(int)
        return X, y

    def train_test_split(self, test_size=0.4, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)


class GISETTE:
    # Reference
    # [1] https://github.com/aashsach/random-forest/blob/master/random_forest_classifier.ipynb

    def __init__(self, root='.'):
        self.X, self.y = self.load_and_preprocess(root)
        self.classification = True
        self.name = 'GISETTE'

    @staticmethod
    def load_and_preprocess(root, verbose=False):
        with open("{}/data/GISETTE/gisette_train.data".format(root)) as f:
            data = []
            for row in f.readlines():
                data.append((row.strip()).split(" "))
        X_train = np.array(data).astype(int)

        with open("{}/data/GISETTE/gisette_train.labels".format(root)) as f:
            classes = []
            for row in f.readlines():
                classes.append((row.strip()).split(" "))
        y_train = np.array(classes).astype(int)
        y_train = y_train[:, 0]

        with open("{}/data/GISETTE/gisette_valid.data".format(root)) as f:
            data = []
            for row in f.readlines():
                data.append((row.strip()).split(" "))
        X_valid = np.array(data).astype(int)

        with open("{}/data/GISETTE/gisette_valid.labels".format(root)) as f:
            classes = []
            for row in f.readlines():
                classes.append((row.strip()).split(" "))
        y_valid = np.array(classes).astype(int)
        y_valid = y_valid[:, 0]

        X = np.concatenate([X_train, X_valid], axis=0)
        y = np.concatenate([y_train, y_valid], axis=0)

        return X, y

    def train_test_split(self, test_size=0.4, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)


