import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, delimiter=",",
                     parse_dates=["Date"]).dropna().drop_duplicates()
    df = df[df["Temp"] > 0]

    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def main():
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df['Country'] == "Israel"]
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year") \
        .write_image("israelDailyTemperatures.png")

    monthly_temp_std = israel_df.groupby(["Month"])['Temp'].std()

    px.bar(x=israel_df['Month'].drop_duplicates(), y=monthly_temp_std,
           title="Monthly Average Temperatures") \
        .update_layout(xaxis_title="Month",
                       yaxis_title="Avg Temperature",
                       title_x=0.5) \
        .write_image("monthTempStd.png")

    # Question 3 - Exploring differences between countries
    gb = df.groupby(["Country", "Month"], as_index=False).agg(
        mean=("Temp", "mean"), std=("Temp", "std"))
    px.line(gb, x="Month", y="mean", error_y="std", color="Country") \
        .write_image("MeanTemperaturesInDifferentCountries.png")

    # Question 4 - Fitting model for different values of `k`
    X = israel_df["DayOfYear"]
    y = israel_df["Temp"]
    pf_result = []
    trainX, trainY, testX, testY = split_train_test(X, y)
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(trainX, trainY)
        pf_result.append(round(pf.loss(testX.to_numpy(), testY.to_numpy()), 2))

    px.bar(x=range(1, 11), y=pf_result,
           title="Test Error As a Function of Degree") \
        .update_layout(xaxis_title="Degree",
                       yaxis_title="TestError",
                       title_x=0.5) \
        .write_image("TestErrorAsAFunctionOfDegree.png")

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    main()
