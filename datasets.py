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
    def load_and_preprocess(root):
        filepath = "{}/data/MiniBooNE/MiniBooNE_PID.txt".format(root)
        df = mb.import_data(Path(filepath))

        # Rename columns
        columns_names = mb.get_column_names(df)
        df.rename(columns=lambda x: columns_names[x], inplace=True)

        # Get the number of elecrons and muons in the file
        num_electron, num_muon = mb.get_num_neutrinos(Path(filepath))

        # Add the "target" column (1 - electron, 0 - muon)
        df = mb.add_target_column(df, num_electron, num_muon)

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

