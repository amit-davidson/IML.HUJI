from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def get_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return np.cov(x, y)[1, 0] / (np.std(x) * np.std(y))


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = X.drop(columns=["id", "date"])
    X["zipcode"] = X["zipcode"].astype(int)
    X = X[X["sqft_living"] > 0]

    for column in ["bathrooms", "floors", "bedrooms"]:
        X = X[X[column] >= 0]

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_arr = y.to_numpy()

    for column in X:
        x_arr = X[column].to_numpy()
        corr = get_pearson_correlation(x_arr, y_arr)
        fig = px.scatter(x=X[column], y=y)
        fig.update_layout(
            title=f"Correlation between {column} and Price<br>Pearson Correlation:{corr}",
            xaxis_title=column,
            yaxis_title="Price",
            title_x=0.5
        )
        fig.write_image(f"CorrelationBetween{column}AndPrice.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - split data into train and test sets
    df = pd.read_csv("../datasets/house_prices.csv",
                     delimiter=",").dropna().drop_duplicates()
    df = df[df["price"] > 0]

    y = df["price"]
    X = df.drop(columns=["price"])

    # Question 2 - Preprocessing of housing prices dataset
    X, y = preprocess_data(X, y)
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    res = np.zeros((91, 10))
    test_x_arr = test_X.to_numpy()
    test_y_arr = test_y.to_numpy()
    p_range = list(range(10, 101))
    for p in p_range:
        for i in range(10):
            sp = train_X.sample(frac=p / 100)
            l = LinearRegression(include_intercept=True)
            l.fit(sp, train_y.loc[sp.index])
            res[p - 10, i] = l.loss(test_x_arr, test_y_arr)

    loss_values = np.mean(res, axis=1)
    print(res)
    fig = px.scatter(x=p_range, y=loss_values)
    fig.update_layout(
        title=f"Loss values as a function of test percentage",
        xaxis_title="Loss value",
        yaxis_title="Test percentage",
        title_x=0.5
    )
    fig.write_image(f"LossValuesAsAFunctionOfTestPercentage.png")

