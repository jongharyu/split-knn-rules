"""
train_model
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune_params(estimator, cv_method: str, param_grid: Dict, x: pd.DataFrame, y: np.array, **kwargs):
    """Runs the given CV method on the given estimator, with the given parameters and their values

    Args:
        estimator: Model (e.g. DecisionTreeClassifier)
        cv_method: CV method ('grid' or 'randomized')
        param_grid: List of parameters and their values
        x: Training X
        y: Training y
        **kwargs: other arguments (e.g. n_jobs)

    Returns:
        CV results
    """

    # Get extra arguments, or set them to default values
    cv = kwargs.get('cv', 3)
    scoring = kwargs.get('scoring', 'roc_auc')
    n_iter = kwargs.get('n_iter', 25)
    return_train_score = kwargs.get('return_train_score', False)
    random_state = kwargs.get('random_state', 123)
    verbose = kwargs.get('verbose', 2)
    n_jobs = kwargs.get('n_jobs', 6)
    refit = kwargs.get('refit', True)

    # Run the given CV method
    if cv_method == "grid":
        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring,
                                   return_train_score=return_train_score,
                                   verbose=verbose,
                                   n_jobs=n_jobs,
                                   refit=refit)
    elif cv_method == "randomized":
        grid_search = RandomizedSearchCV(estimator=estimator,
                                         param_distributions=param_grid,
                                         n_iter=n_iter,
                                         cv=cv,
                                         scoring=scoring,
                                         random_state=random_state,
                                         return_train_score=return_train_score,
                                         verbose=verbose,
                                         n_jobs=n_jobs,
                                         refit=refit)
    else:
        raise Exception("Please enter 'grid' or 'randomized' as cv_method!")

    # Fit
    grid_search.fit(x, y)

    print("The best parameters: {}".format(grid_search.best_params_))
    print("The best score: {}".format(grid_search.best_score_))

    return grid_search


def plot_cv_results(cv_results: Dict) -> None:
    """Plot the scores for the searched parameters

    Args:
        cv_results: Cross-validation results

    Returns:
        None
    """
    # Get scores
    mean_test_scores = cv_results['mean_test_score']

    # Get list of parameters
    params = cv_results['params']

    # Put parameters and scores into a DataFrame for easy plotting
    df = pd.DataFrame(params)
    df['score'] = mean_test_scores

    # Obtain the top score
    top_value = df.nlargest(n=1, columns='score')['score'].max()
    top_df = df.loc[df['score'] == top_value]

    # Graph
    for index, col in enumerate([c for c in df.columns if c not in ['is_top', 'score']]):
        grid = sns.jointplot(data=df, x=col, y='score')
        grid.fig.set_figwidth(15)
        grid.fig.set_figheight(5)
        grid.x = top_df[col]
        grid.y = top_df['score']
        # grid.plot_join(plt.scatter, marker='x', c='g', s=50)

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def evaluate_params(estimator, cv_method: str, param_grid: Dict, x: pd.DataFrame, y: np.array, **kwargs):
    """ Wrapper function for tuning and plotting parameter searching results

    Args:
        estimator: Model (e.g. DecisionTreeClassifier)
        cv_method: CV method ('grid' or 'randomized')
        param_grid: List of parameters and their values
        x: Training X
        y: Training y
        **kwargs: other arguments (e.g. n_jobs)

    Returns:
        CV results

    """

    # Tune parameters
    grid_search = tune_params(estimator=estimator,
                              cv_method=cv_method,
                              param_grid=param_grid,
                              x=x,
                              y=y,
                              **kwargs)

    # Plot results
    plot_cv_results(cv_results=grid_search.cv_results_)

    return grid_search
