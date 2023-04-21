import sys
from typing import Tuple

import numpy as np

sys.path.append("../")
from IMLearn.utils import utils as ut
from IMLearn.learners.regressors import linear_regression
from utils import *
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from exercises.house_price_prediction import preprocess_data
import pandas as pd
from typing import Tuple


def create_dummy_house_dataframe(house_data) -> \
        Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(np.delete(np.array(house_data), 2, 1),
                        columns=['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                                 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                                 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                 'sqft_living15', 'sqft_lot15'])
    X['bedrooms'] = X['bedrooms'].astype('float')
    X['sqft_lot'] = X['sqft_lot'].astype('float')
    X['bathrooms'] = X['bathrooms'].astype('float')
    X['yr_built'] = X['yr_built'].astype('float')
    X['floors'] = X['floors'].astype('float')
    X['sqft_living'] = X['sqft_living'].astype('float')
    X['zipcode'] = X['zipcode'].astype('float')
    y = pd.Series(data=np.array(house_data)[:, 2]).astype('float')
    return X, y

def test_preprocess_irrelevant_cols_deleted():
    irrelevant_cols = {'id', 'sqft_living15', 'sqft_lot15'}  # Change according to your implementation
    # Arrange
    X, y = create_dummy_house_dataframe(
        [["7129300520", "20141013T000000", 221900, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178",
          47.5112,
          -122.257, 1340, 5650],
         ["6414100192", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125",
          47.721, -122.319, 1690, 7639]])

    # Act
    X, y = preprocess_data(X, y)

    # Assert
    assert len(irrelevant_cols.intersection(X.columns)) == 0


def test_preprocess_bad_rows_negative_values():
    # Arrange
    non_negative_X_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'sqft_above',
                           "sqft_basement", "yr_built", "yr_renovated"]  # Change according to your implementation
    X, y = create_dummy_house_dataframe(
        [["1", "20141013T000000", 10000, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["1", "20141013T000000", -1, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["2", "20141209T000000", 538000, -1, 2.25, 2570, 7242, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125",
          47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, -1, 2570, 7242, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991,
          "98125",
          47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, 2.25, -1, 7242, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, 2.25, 2570, -1, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "-1", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "-1", -1, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "-1", 0, -1, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["3", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "-1", 0, -0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639]])

    # Act
    X, y = preprocess_data(X, y)

    # Assert
    assert np.all(y.astype(int) >= 0)
    assert not (X[non_negative_X_cols].astype(float) < 0).any().any()


def test_preprocess_bad_rows_NaN():
    # Arrange
    non_negative_X_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'sqft_above',
                           "sqft_basement", "yr_built", "yr_renovated", "zipcode"]
    X, y = create_dummy_house_dataframe(
        [["1", "20141013T000000", 10000, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["2", "20141013T000000", np.nan, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         [np.nan] * 21])

    # Act
    X, y = preprocess_data(X, y)

    # Assert
    assert not (X.isna().any().any() or y.isna().any().any())


def test_preprocess_hard_null_check():
    # Arrange
    bad_values = ['NA', 'N/A', None, np.nan]
    X, y = create_dummy_house_dataframe(
        [["1", "20141013T000000", 10000, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["2", "20141013T000000", np.nan, np.nan, np.nan, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["2", "20141013T000000", 10000, None, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["2", "20141013T000000", 10000, 3, 'NA', 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["2", "20141013T000000", 10000, 3, 1, 'N/A', 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650]])

    # Act
    X = preprocess_data(X)

    # Assert - check if any bad value is anywhere in the DataFrame
    bad_value_in_X = False
    for val in bad_values:
        if pd.isna(val):
            bad_value_in_X = bad_value_in_X or X.isna().values.any()
        else:
            bad_value_in_X = bad_value_in_X or X.eq(val).any().any()
    assert not bad_value_in_X


def test_preprocess_when_bad_values_in_irrelevant_columns_should_keep_row():
    # Arrange
    X, y = create_dummy_house_dataframe(
        [[np.NAN, "20141013T000000", 10000, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650]])

    # Act
    X, y = preprocess_data(X, y)

    # Assert
    assert len(X) == len(y) == 1


def test_preprocess_when_bad_row_should_remove_from_both_X_and_y():
    # Arrange
    non_negative_X_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'sqft_above',
                           "sqft_basement", "yr_built", "yr_renovated", "zipcode"]
    X, y = create_dummy_house_dataframe(
        [["1", "20141013T000000", 10000, 3, 1, -1, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650]])

    # Act
    X, y = preprocess_data(X, y)

    # Assert
    assert len(X) == len(y) == 0


def test_preprocess_categorical_features():
    # Arrange
    X, y = create_dummy_house_dataframe(
        [["7129300520", "20141013T000000", 221900, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["6414100192", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["5631500400", "20150225T000000", 180000, 2, 1, 770, 10000, "1", 0, 0, 3, 6, 770, 0, 1933, 0, "98028", 47.7379, -122.233, 2720, 8062]])
    # Act
    X = preprocess_data(X)

    # Assert
    assert 'zipcode' not in X.columns  # add more categorical features according to your implementation


def test_preprocess_date_should_be_numerical():
    # Arrange
    dates = ["20141013T000000", "20141209T000000", "20150225T000000"]
    X, y = create_dummy_house_dataframe(
        [["7129300520", "20141013T000000", 221900, 3, 1, 1180, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["6414100192", "20141209T000000", 538000, 3, 2.25, 2570, 7242, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["5631500400", "20150225T000000", 180000, 2, 1, 770, 10000, "1", 0, 0, 3, 6, 770, 0, 1933, 0, "98028", 47.7379, -122.233, 2720, 8062]])
    # Act
    X = preprocess_data(X)

    # Assert - check date was changed in some way (can't check how)
    assert not X['date'].isin(dates).any().any()


def test_split(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75):
    if X.empty:
        # Call should succeed
        ut.split_train_test(X, y, proportion)
        return

    # Act
    train_x, train_y, test_x, test_y = ut.split_train_test(X, y, proportion)

    # Assert
    # Split should be by given formula
    assert train_x.shape[0] == np.ceil(proportion * X.shape[0])
    assert train_y.shape[0] == np.ceil(proportion * y.shape[0])
    assert test_x.shape[0] == np.floor((np.around(1 - proportion, 2)) * X.shape[0])
    assert test_y.shape[0] == np.floor(np.around(1 - proportion, 2) * y.shape[0])

    # Reconstruction of X and y from test and train parts should succeed
    pd.testing.assert_frame_equal(pd.concat([train_x, test_x]).sort_index(), X)
    pd.testing.assert_frame_equal(pd.concat([train_y.to_frame(), test_y.to_frame()]).sort_index(), y.to_frame())


def test_lin_reg_samples_from_line_test(epsilon: float = 0.000001):
    global y
    print("Simple test - Find the straight line y = x + 1")
    # Arrange
    x = np.array([0, 1, 2, 3]).T
    y = np.array([1, 2, 3, 4]).T
    w = np.array([1, 1])

    # Act
    lre = linear_regression.LinearRegression(include_intercept=True)
    lre.fit(x, y)

    # Assert
    assert lre.coefs_.shape == w.shape
    assert np.all(np.abs(lre.coefs_ - w) < epsilon)

    # Arrange
    x = x + 1
    correct_y = y + 1

    # Act
    prediction = lre.predict(x)

    # Assert
    assert prediction.shape == correct_y.shape
    assert np.all(np.abs(prediction - correct_y) < epsilon)
    print(f"PASSED\n")


def test_lin_reg_compete_with_sklearn(intercept: bool = True, epsilon: float = 0.000001):
    global y
    print(f"Harder test - Compare results with sklearn model. intercept={intercept}")
    # Arrange
    # Create some correlated data
    cov = np.array([[1, 0.8, .7, .6], [.8, 1., .5, .5], [0.7, .5, 1., .5], [0.6, .5, .5, 1]])
    scores_train = mvn.rvs(mean=[80., 80., 80., 80.], cov=cov, size=1000)
    X = pd.DataFrame(data=scores_train[:, :3], columns=["Math", "Science", "History"])
    y = pd.DataFrame(data=scores_train[:, 3:], columns=["Art"])  # Try to predict score in Art from other subjects
    scores_test = mvn.rvs(mean=[80., 80., 80., 80.], cov=cov, size=1000)
    X_test = pd.DataFrame(data=scores_test[:, :3], columns=["Math", "Science", "History"])
    y_test = pd.DataFrame(data=scores_test[:, 3:], columns=["Art"])  # Try to predict score in Art from other subjects

    # Act
    sklearn_learner = LinearRegression(fit_intercept=intercept)
    sklearn_learner.fit(X, y)
    your_learner = linear_regression.LinearRegression(include_intercept=intercept)
    your_learner.fit(X.to_numpy(), y.to_numpy())
    y_hat_by_sklearn = sklearn_learner.predict(X_test)
    y_hat_by_you = your_learner.predict(X_test.to_numpy())

    # Assert
    assert y_hat_by_you.shape == y_hat_by_sklearn.shape
    assert np.all(np.abs(y_hat_by_you - y_hat_by_sklearn) < epsilon)
    assert np.abs(your_learner._loss(X_test.to_numpy(), y_test.to_numpy())
                  - mean_squared_error(y_hat_by_sklearn, y_test)) < epsilon
    print(f"PASSED\n")


def test_split_train_test():
    global proportion
    print("Testing split_train_test with various input sizes and proportions\n")
    # Arrange
    main_X = pd.DataFrame(np.array(
        [[1, 2],
         [4, 5],
         [7, 8],
         [10, 11],
         [13, 14]]))
    main_y = pd.Series(np.array([3, 6, 9, 12, 15]))
    for input_size in range(0, main_X.shape[0] + 1):
        for proportion in np.linspace(0.1, 1, 10):
            print(f"Testing with input size: {input_size}, proportion:{proportion}")
            test_split(main_X[:input_size], main_y[:input_size], proportion)
            print(f"PASSED\n")


def test_preprocess_bad_rows_tiny_rooms():
    # Arrange
    sqft_rooms = ['sqft_living', 'sqft_lot']
    minimum_size_in_sqft = 35  # Change according to your implementation
    X, y = create_dummy_house_dataframe(
        [["7129300520", "20141013T000000", 221900, 3, 1, 20, 5650, "1", 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["6414100192", "20141209T000000", 538000, 3, 2.25, 2570, 20, "2", 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["5631500400", "20150225T000000", 180000, 2, 1, 770, 10000, "1", 0, 0, 3, 6, 770, 0, 1933, 0, "98028", 47.7379, -122.233, 2720, 8062]])
    # Act
    X, _ = preprocess_data(X)

    # Assert
    assert not (X[sqft_rooms].astype(float) < minimum_size_in_sqft).any().any()


def test_preprocess_when_only_X_provided_should_succeed():
    # Arrange
    X, y = create_dummy_house_dataframe(
        [["7129300520", "20141013T000000", 221900, 3, 1, 500, 5650, 1, 0, 0, 3, 7, 1180, 0, 1955, 0, "98178", 47.5112, -122.257, 1340, 5650],
         ["6414100192", "20141209T000000", 538000, 3, 2.25, 2570, 500, 2, 0, 0, 3, 7, 2170, 400, 1951, 1991, "98125", 47.721, -122.319, 1690, 7639],
         ["5631500400", "20150225T000000", 180000, 2, 1, 770, 10000, 1, 0, 0, 3, 6, 770, 0, 1933, 0, "98028", 47.7379, -122.233, 2720, 8062]])
    # Act
    X, _ = preprocess_data(X.assign(price=y))  # Call should succeed and return a dataframe

    # Assert
    assert 'price' in X.columns


if __name__ == '__main__':
    print("Starting tests - good luck!")
    print("################################# Test linear_regression #################################\n")
    test_lin_reg_samples_from_line_test()
    test_lin_reg_compete_with_sklearn(intercept=True)
    test_lin_reg_compete_with_sklearn(intercept=False)

    print("################################# Test Utils #################################\n")
    # test_split_train_test()

    print("################################# Test house_price_prediction #################################\n")
    print("#### Testing preprocess_data ####\n")
    test_preprocess_when_only_X_provided_should_succeed()
    test_preprocess_irrelevant_cols_deleted()
    # test_preprocess_bad_rows_negative_values()
    test_preprocess_bad_rows_tiny_rooms()
    test_preprocess_bad_rows_NaN()
    test_preprocess_hard_null_check()
    test_preprocess_when_bad_values_in_irrelevant_columns_should_keep_row()  # Assumes id is irrelevant. Turn off if you implemented differently
    test_preprocess_when_bad_row_should_remove_from_both_X_and_y()  # Assumes rows with bad values are deleted. Turn off if you implemented differently
    test_preprocess_categorical_features()
    test_preprocess_date_should_be_numerical()

    print("#### Testing feature_evaluation ####\n")

    print(f"PASSED\n")
    print("\nPASSED ALL TESTS")
