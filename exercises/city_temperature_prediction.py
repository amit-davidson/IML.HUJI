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
    df["MonthNum"] = df["Date"].dt.month
    return df


def main():
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df['Country'] == "Israel"]
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year") \
        .write_image("israelDailyTemperatures.png")

    monthly_temp_std = israel_df.groupby(["MonthNum"])['Temp'].std()

    px.bar(x=israel_df['MonthNum'].drop_duplicates(), y=monthly_temp_std,
           title="Monthly Average Tempartures") \
        .update_layout(title="Average Monthly Temperatures",
                       xaxis_title="Month",
                       yaxis_title="Avg Temperature",
                       title_x=0.5) \
        .write_image("monthTempStd.png")

    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    main()
