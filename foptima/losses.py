import numpy as np


def chi_squared(y_true, y_est, weights):
    return np.mean(weights * (y_true - y_est) ** 2)


_losses = {
    "chi_squared": chi_squared,
    "chi2": chi_squared,
}
