"""
make_dataset
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


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
