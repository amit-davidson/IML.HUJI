from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def get_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return np.cov(x, y)[1, 0] / (np.std(x) * np.std(y))


avg_values = []
columns_dummies = []


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
    global avg_values, columns_dummies
    # Train
    if y is not None:
        df = X.assign(price=y.values)
        df = df[df["bedrooms"] < 15]
        df = df[df["sqft_lot"] < 10000000]

        for column in ["bathrooms", "floors", "bedrooms"]:
            if column in df.columns:
                df = df[df[column] >= 0]
        #
        for column in ["yr_built", "zipcode", "price", "sqft_living"]:
            if column in X.columns:
                df = df[df[column] > 0]
        #
        df = pd.get_dummies(df, prefix="zipcode", columns=["zipcode"])
        y = df["price"]
        X = df.drop(columns=["price"])
        avg_values = X.mean()
        columns_dummies = X.columns
        return X, y
    # Test
    else:
        X = X.fillna(avg_values)
        X = pd.get_dummies(X, prefix="zipcode", columns=["zipcode"])
        X = X.reindex(columns=columns_dummies, fill_value=0)
        return X, None


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
        if column.startswith("zipcode"):
            continue
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


def main():
    # Question 1 - split data into train and test sets
    df = pd.read_csv("../datasets/house_prices.csv",
                     delimiter=",").drop_duplicates()
    df = df.drop(columns=["id", "date", "sqft_living15", "sqft_lot15"])

    df = df.dropna(subset=['price'])

    y = df["price"]
    X = df.drop(columns=["price"])

    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_X, _ = preprocess_data(test_X, None)

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
    std_values = np.std(res, axis=1)

    scatter = [
        go.Scatter(x=p_range, y=loss_values, mode="markers+lines",
                   name="Mean loss", showlegend=False),
        go.Scatter(x=p_range, y=loss_values + 2 * std_values, mode="lines",
                   name="Mean loss", fill="tonexty",
                   line=dict(color="lightgrey"), showlegend=False),
        go.Scatter(x=p_range, y=loss_values - 2 * std_values, mode="lines",
                   name="Mean loss", fill="tonexty",
                   line=dict(color="lightgrey"), showlegend=False),
    ]
    layout = go.Layout(
        title={"text": "Loss values as a function of test percentage",
               "x": 0.5},
        xaxis={"title": "Test percentage %p"},
        yaxis={"title": "Loss value"},
    )
    fig = go.Figure(data=scatter, layout=layout)
    fig.write_image(f"LossValuesAsAFunctionOfTestPercentage.png")


if __name__ == '__main__':
    np.random.seed(0)
    main()
