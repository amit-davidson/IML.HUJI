from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, size = 10, 1, 1000
    dist = np.random.normal(mu, sigma, size)
    model = UnivariateGaussian().fit(dist)
    print((model.mu_, model.var_))


    # Question 2 - Empirically showing sample mean is consistent
    y_axis = [np.abs(mu-UnivariateGaussian().fit(dist[:n]).mu_) for n in np.arange(10, 1000, 10)]
    fig = px.scatter(x=list(range(len(y_axis))), y=y_axis)
    fig.update_layout(
        title="Deviation of Sample Mean As a Function of Sample Size",
        xaxis_title="Sample Size",
        yaxis_title="Deviation Mean",
        title_x=0.5
    )
    fig.write_image("meanDeviationAgainstSampleSize.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    y_axis = model.pdf(dist)
    fig = px.scatter(x=dist, y=y_axis)
    fig.update_layout(
        title="PDF of our model",
        xaxis_title="Sample Value",
        yaxis_title="Sample Result",
        title_x=0.5
    )
    fig.write_image("ModelPDF.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
