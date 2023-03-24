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
    y_axis = [np.abs(mu - UnivariateGaussian().fit(dist[:n]).mu_) for n in
              np.arange(10, 1000, 10)]
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
        title="PDF of a Univariate Distribution",
        xaxis_title="Sample Value",
        yaxis_title="Sample Result",
        title_x=0.5
    )
    fig.write_image("UnivariateModelPDF.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = np.array(
        [[1, 0.2, 0, .5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]]
    )
    size: int = 1000

    samples = np.random.multivariate_normal(mu, cov, size)
    model = MultivariateGaussian().fit(samples)
    print(np.round(model.mu_, 3))
    print(np.round(model.cov_, 3))

    # Question 5 - Likelihood evaluation
    res = np.zeros((200, 200))
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    for i in range(200):
        for j in range(200):
            res[i, j] = MultivariateGaussian.log_likelihood(
                np.array([f1[i], 0, f3[j], 0]), cov, samples)

    fig = px.imshow(res, x=f3, y=f1)
    fig.update_layout(
        title="Log-Likelihood Multivatiate Gaussian As Function of Features f1"
              " and f3",
        xaxis_title="f3",
        yaxis_title="f1",
        title_x=0.5
    )
    fig.write_image("MultivatiateModelPDF.png")

    # Question 6 - Maximum likelihood
    i, j = np.unravel_index(res.argmax(), res.shape)
    print(np.round(f1[i], 3), np.round(f3[j], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
