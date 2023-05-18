import numpy as np


EPSILON = 1e-8


def chi_squared(y_true, y_est, weights):
    return np.mean(weights * (y_true - y_est) ** 2)


def laplace(y_true, y_est, weights):
    return np.mean(weights * np.abs(y_true - y_est))


def poisson(y_true, y_est, weights):
    return np.mean(weights * (y_est - y_true * np.log(y_est + EPSILON)))



_losses = {
    "chi_squared": chi_squared,
    "chi2": chi_squared,
    "mse": chi_squared,
    "l2": chi_squared,
    "laplace": laplace,
    "l1": laplace,
    "poisson": poisson,
}
