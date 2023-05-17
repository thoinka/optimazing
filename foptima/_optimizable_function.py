from typing import Callable, Optional, Dict, Any, Union, Tuple, Iterable
from inspect import getfullargspec
import numpy as np
from scipy.optimize import minimize
from ._optimization_result import OptimizationResult
from .losses import _losses


def optimizable(function):
    return OptimizableFunction(function)


class OptimizableFunction:
    """Optimizable function with methods `fit`, `freeze` and `bound`.

    Parameters
    ----------
    function: callable
        Function with parameters as keyword-only arguments.
    freeze_dict: dict(str, float)
        Dictionary that maps frozen parameters to their values.
    bounds: dict(str, tuple(float, float))
        Dictionary that maps bounded parameters to its bounds. A `None` bound is
        open.
    """

    def __init__(
        self,
        function: Callable,
        freeze_dict: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self._function = function
        self._freeze_dict = freeze_dict or {}
        self._bounds_dict = bounds or {}
        argspecs = getfullargspec(function)
        self._check_param_and_arg_names(argspecs.kwonlyargs, argspecs.args)
        self._arguments = argspecs.args
        self._parameters = argspecs.kwonlyargs
        self._name = function.__name__

    def _check_param_and_arg_names(self, params, args):
        for a in args:
            if a.startswith("_"):
                raise ValueError(
                    f"Arguments cannot begin with underscore, but found {a}"
                )
        for p in params:
            if p.startswith("_"):
                raise ValueError(
                    f"Arguments cannot begin with underscore, but found {a}"
                )

    @property
    def num_args(self) -> int:
        return len(self._arguments)

    @property
    def num_params(self) -> int:
        return len(self._parameters)

    @property
    def is_bounded(self) -> bool:
        return len(self._bounds_dict) > 0

    @property
    def is_frozen(self) -> bool:
        return len(self._freeze_dict) > 0

    def _check_parameters(self, kwargs):
        for key in kwargs.keys():
            if key not in self._parameters:
                raise ValueError(f"Parameter {key} unknown.")
            if key in self._freeze_dict:
                raise ValueError(f"Specified frozen parameter {key}.")

    def __call__(self, *args, **kwargs):
        self._check_parameters(kwargs)
        return self._function(*args, **kwargs, **self._freeze_dict)

    def freeze(self, **kwargs):
        """Freezes one or more parameters of an optimizable function.

        Parameters
        ----------
        kwargs: dict
            Dictionary of frozen parameters with respective values.

        Returns
        -------
        OptimizableFunction

        Example
        -------
        The following would freeze one parameter of a linear function to zero so that
        it can be fit without a y-offset:
        >>> @optimizable
        ... linear(x, *, a, b):
        ...     return a * x + b
        ...
        ... linear.freeze(b=0.0)
        """
        self._check_parameters(kwargs)
        return OptimizableFunction(
            self._function, {**self._freeze_dict, **kwargs}, self._bounds_dict
        )

    def bound(self, **kwargs):
        """Adds bounds to one or more parameters of an optimizable function.

        Parameters
        ----------
        kwargs: dict
            Dictionary of frozen parameters with respective bounds as a tuple like
            (lower, upper).

        Returns
        -------
        OptimizableFunction

        Example
        -------
        The following would bound the slope of a linear function to be positive:
        >>> @optimizable
        ... linear(x, *, a, b):
        ...     return a * x + b
        ...
        ... linear.bound(a=(0.0, None))
        """
        self._check_parameters(kwargs)
        return OptimizableFunction(
            self._function, self._freeze_dict, {**self._bounds_dict, **kwargs}
        )

    def fit(
        self,
        args: Union[Iterable[Iterable[float]], Iterable[float]],
        y: Iterable[float],
        loss: Union[str, Callable] = "chi2",
        w: Optional[Iterable[float]] = None,
        options: Dict[str, Any] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Fits all free parameters of an optimizable function.

        Parameters
        ----------
        args: iterable(float) or iterable(iterable(float))
            x-values for the fit. If multiple inputs exist, then this has to be some
            iterable over those.
        y: iterable(float)
            Target values.
        w: iterable(float)
            Weights.
        loss: string or callable
            Loss function. If string, has to be one of:
            * "mse"
            * "l1"
            * ...
        options: dict
            Additional options. Will be propagated to scipy.optimize.minimize
        kwargs: dict
            Initial values for fit.

        Returns
        -------
        OptimizationResult

        Example
        -------
        The following would fit a linear function
        >>> @optimizable
        ... linear(x, *, a, b):
        ...     return a * x + b
        ...
        ... linear.fit([0, 1, 2], [0.123, 0.938, 2.123], m=1, b=0)
        """
        args = np.array(args)
        if args.ndim == 1:
            args = args[None]

        w = w or np.ones(len(y))

        if isinstance(loss, str):
            loss = _losses[loss.lower()]

        free_parameters = [p for p in self._parameters if p not in self._freeze_dict]
        if len(free_parameters) == 0:
            raise ValueError("Attempted to fit, but no free parameters left!")

        def _optimization_function(p):
            params = {key: value for key, value in zip(free_parameters, p)}
            params.update(self._freeze_dict)
            return loss(y, self._function(*args, **params), w)

        opt_config = {
            "x0": [kwargs.get(p, 1.0) for p in free_parameters],
        }
        opt_config.update(options or {})
        if self.is_bounded:
            opt_config.update(
                bounds=[self._bounds_dict.get(p, (None, None)) for p in free_parameters]
            )
        result = minimize(_optimization_function, **opt_config)
        values = {p: v for p, v in zip(free_parameters, result.x)}
        values.update(self._freeze_dict)
        if hasattr(result, "hess_inv") and hasattr(result.hess_inv, "diagonal"):
            _unc = np.sqrt(result.hess_inv.diagonal())
            uncertainties = {p: v for p, v in zip(free_parameters, _unc)}
        else:
            uncertainties = {p: None for p in free_parameters}
        uncertainties.update({p: None for p in self._freeze_dict})
        return OptimizationResult(self._function, values, uncertainties)

    def __repr__(self):
        args = ", ".join(self._arguments)
        params = ", ".join(self._parameters)
        return f"<OptimizableFunction {self._name}({args}; {params})>"
