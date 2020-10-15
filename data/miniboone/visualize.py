"""
visualize
"""

import matplotlib.pyplot as plt
import pandas as pd


def pretty_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty describe a DataFrame

    Args:
        df: DataFrame

    Returns:
        Description in a DataFrame format
    """
    return pd.DataFrame(df.describe(percentiles=[.10, .25, .5, .75, .90]).T) \
        .applymap("{0:,.3f}".format)


def plot_roc_curve(fpr, tpr, label=None):
    """
    Plot ROC AUC curves

    Args:
        fpr: Array of false positive rates
        tpr: Array of true positive rates
        label: Label (e.g. "Training")

    Returns:
        None
    """
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
