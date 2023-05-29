"""
Losses. A loss has to be a function
loss(
    y_true: np.ndarray,
    y_est: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray
) -> float
"""
import numpy as np
from inspect import getfullargspec
from abc import ABC
from typing import Callable, Optional, Dict


class _DictWithGetAttr:
    def __init__(self, obj: Optional[Dict] = None):
        self.__dict__.update(obj or {})

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def __repr__(self):
        return "\n".join([f"{k}: {repr(v)}" for k, v in self.__dict__.items()])


_losses = _DictWithGetAttr()


def loss(func: Optional[Callable] = None, *, register: Optional[bool] = False):
    def _decorator(func: Callable):
        """Turns a function into a BaseLoss"""
        argspecs = getfullargspec(func)
        if argspecs.args != ["y_true", "y_est", "weights", "sigma"]:
            raise ValueError(
                f"Loss function {func.__name__} has to have the following signature: "
                f"loss(y_true, y_est, weights, sigma)"
            )

        class Loss(BaseLoss):
            def __init__(self, **kwargs):
                super().__init__(func.__name__, **kwargs)

            def function(self, y_true, y_est, weights, sigma):
                return func(y_true, y_est, weights, sigma)

        _loss = Loss()
        if register:
            _losses[_loss.name] = _loss
        return _loss

    if func is None:
        return _decorator
    return _decorator(func)


class BaseLoss(ABC):
    def __init__(self, name, **params):
        self.name = name
        self.params = params

    def _return_with_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def function(self, y_true, y_est, weights, sigma):
        raise NotImplementedError

    def __call__(self, y_true=None, y_est=None, weights=None, sigma=None, **kwargs):
        if y_true is None and y_est is None and weights is None and sigma is None:
            return self._return_with_params(**kwargs)
        return self.function(y_true, y_est, weights, sigma, **self.params)

    def __repr__(self):
        return f"<loss {self.name}(y_true, y_est, weights, sigma)>"


@loss(register=True)
def chi_squared(y_true, y_est, weights, sigma):
    return np.mean(weights * (y_true - y_est) ** 2 / sigma**2)


@loss(register=True)
def laplace(y_true, y_est, weights, sigma):
    return np.mean(weights * np.abs(y_true - y_est) / sigma)


@loss(register=True)
def poisson(y_true, y_est, weights, sigma, *, epsilon: float = 1e-8):
    return np.mean(weights * (y_est - y_true * np.log(y_est + epsilon)))
