# Reference: https://github.com/david-siqi-liu/miniboone/
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Tuple


def import_data(fpath: Path) -> pd.DataFrame:
    """Import the MiniBooNE data text file

    Args:
        fpath: path to the text file

    Returns:
        DataFrame

    """
    if fpath.suffix != ".txt":
        raise TypeError("Please ensure the file is a text file!")

    if not fpath.exists():
        raise FileNotFoundError("{0} does not exist!".format(fpath.resolve()))

    with open(fpath, "r") as fstream:
        return pd.read_csv(fstream
                           , delim_whitespace=True
                           , skiprows=1
                           , header=None)


def get_column_names(df: pd.DataFrame) -> List[str]:
    """Get number of particles from the DataFrame, and return a list of column names

    Args:
        df: DataFrame

    Returns:
        List of columns (e.g. PID_xx)

    """
    c = df.shape[1]

    if c <= 0:
        raise IndexError("Please ensure the DataFrame isn't empty!")

    return ["PID_{0}".format(x + 1) for x in range(c)]


def get_num_neutrinos(fpath: Path) -> Tuple[int, int]:
    """Get the number of neutrinos (both electrons and muons) from the MiniBooNE data text file

    Args:
        fpath: path to the text file

    Returns:
        Tuple of (num_electron, num_muon)

    """
    if fpath.suffix != ".txt":
        raise TypeError("Please ensure the file is a text file!")

    if not fpath.exists():
        raise FileNotFoundError("{0} does not exist!".format(fpath.resolve()))

    line = open(fpath).readline().split()

    try:
        num_electron = int(line[0])
        num_muon = int(line[1])
    except ValueError:
        raise ValueError("Please ensure the first line contains two integers!")

    # print("Number of electrons: {0:,}\nNumber of muons: {1:,}".format(num_electron, num_muon))
    return num_electron, num_muon


def add_target_column(df: pd.DataFrame, num_electron: int, num_muon: int) -> pd.DataFrame:
    """Assign target column to the given DataFrame, based on the number of neutrinos given

    Args:
        df: DataFrame of the data
        num_electron: number of electrons
        num_muon: number of muons

    Returns:
        DataFrame with "target" column added
    """
    if 'target' in df.columns:
        raise Exception("'target' column already exists.")

    if num_electron + num_muon != df.shape[0]:
        raise Exception("Row numbers do not match!\nRows:{0:,}\nElectrons:{1:,}\nMuons:{2:,}"
                        .format(df.shape[0], num_electron, num_muon))

    # Initialize with NULL values
    df['target'] = np.nan

    # Assign values
    df['target'][:num_electron] = 1
    df['target'][num_electron:] = 0

    # Downcast to int8
    df['target'] = pd.to_numeric(df['target'], downcast='integer')

    return df


def train_test_stratified_split(df: pd.DataFrame,
                                target: str,
                                test_size: float = 0.2,
                                random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified sampling of the DataFrame into training and test sets

    Args:
        df: DataFrame that is to be split
        target: name of the target column
        test_size: test size, default is 0.2
        random_state: random seed, default is 0

    Returns:
        Tuple of training and test DataFrames
    """

    if target not in df.columns:
        raise IndexError("Column '{0}' is not in the DataFrame!".format(target))

    split = StratifiedShuffleSplit(n_splits=2,
                                   test_size=test_size,
                                   random_state=random_state)

    for train_index, test_index in split.split(df, df[target]):
        # print("Training set size: {0:,}\nTest set size: {1:,}".format(train_index.size, test_index.size))
        return df.loc[train_index], df.loc[test_index]


def x_y_split(df: pd.DataFrame,
              y_col: str) -> Tuple[pd.DataFrame, np.array]:
    """
    Split a DataFrame into x (explanatory variables) and y (target variable)

    Args:
        df: DataFrame that is to be split
        y_col: name of the y column

    Returns:
        Tuple of x (DataFrame) and y (array)
    """

    if y_col not in df.columns:
        raise IndexError("Column '{0}' is not in the DataFrame!".format(y_col))

    x = df.drop(y_col, axis=1).copy()
    y = df[y_col].copy()

    # print("Class ratio: {0:.5f}".format(y.sum() / x.shape[0]))

    return x, y

def pretty_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty describe a DataFrame

    Args:
        df: DataFrame

    Returns:
        Description in a DataFrame format
    """
    return pd.DataFrame(df.describe(percentiles=[.10, .25, .5, .75, .90]).T) \
        .applymap("{0:,.3f}".format)
