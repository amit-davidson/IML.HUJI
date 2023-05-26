import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise=0, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab = AdaBoost(lambda: DecisionStump(), n_learners)
    ab.fit(train_X, train_y)

    train_losses = list()
    test_losses = list()
    for i in range(1, n_learners + 1):
        train_losses.append(ab.partial_loss(train_X, train_y, i))
        test_losses.append(ab.partial_loss(test_X, test_y, i))

    fig = px.line()
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_losses,
                             mode='markers'))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=train_losses,
                             mode='markers'))

    fig.update_layout(
        title=f"Error of AdaBoost using DecisionStump of train and test sets as a function of iteration count",
        xaxis_title="Iteration",
        yaxis_title="Error",
        title_x=0.5,
    )
    fig.write_image(f"AdaBoostErrorAsAFunctionOfIterations.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost()
