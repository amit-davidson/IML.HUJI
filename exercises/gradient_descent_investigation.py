import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from utils import custom
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly.express as px


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weightss, vals = list(), list()
    def cb(solver: GradientDescent, weights: np.ndarray, val: np.ndarray, grad: np.ndarray, t: int, eta: float, delta: float):
        weightss.append(weights)
        vals.append(val)

    return cb, vals, weightss


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for m in [L1, L2]:
        for eta in etas:
            callback, vals, w = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(m(weights=init), np.array([]), np.array([]))
            fig = plot_descent_path(m, np.array(w), title=f"Descent path of {m.__name__} Module eta={eta}")
            fig.write_image(f"descent_path_{m.__name__}_{eta}.png")

            fig = px.scatter(x=list(range(len(vals))), y=vals)
            fig.update_layout(
                title=f"convergence rate {m.__name__} Module eta={eta}",
                xaxis_title="Iteration",
                yaxis_title="Error",
                title_x=0.5,
            )
            fig.write_image(f"convergence_rate_{m.__name__}_{eta}.png")
            print(f"{m.__name__}: {format(min(vals), '.10f')}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train),\
                                       np.array(X_test), np.array(y_test)

    callback, _, _ = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=FixedLR(1e-4), max_iter=20000)

    lr = LogisticRegression(solver=gd)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    c = [custom[0], custom[-1]]
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.write_image("ROC_model.png")

    best_alpha = np.round(thresholds[np.argmax(tpr - fpr)], decimals=2)
    print("Best alpha:", best_alpha)
    lr_best_alpha = LogisticRegression(solver=gd, alpha=best_alpha)
    lr_best_alpha.fit(X_train, y_train)
    print("Loss: ", np.round(lr_best_alpha.loss(X_test, y_test), decimals=2))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lams = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
    for module in ("l1", "l2"):
        validation_errors = list()
        for lam in lams:
            lr_reg = LogisticRegression(solver=gd, alpha=0.5, penalty=module,
                                       lam=lam)
            _, validation_err = cross_validate(lr_reg, X_train, y_train,
                misclassification_error, 5)
            validation_errors.append(validation_err)

        best_lam = lams[int(np.argmin(validation_errors))]
        lr = LogisticRegression(solver=gd, penalty=module, alpha=0.5,
                                lam=best_lam)
        lr.fit(X_train, y_train)
        print(f"{module}:")
        print(f"Best lambda:{best_lam}")
        print(f"Error: {np.round(lr.loss(X_test, y_test), decimals=2)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
