import numpy as np
from typing import Optional, Callable, Dict
from inspect import getfullargspec


class ParameterValue:
    __slots__ = ["value", "uncertainty"]

    def __init__(self, value: float, uncertainty: Optional[float] = None):
        self.value = value
        self.uncertainty = uncertainty

    def __str__(self):
        out = f"{self.value}"
        if self.uncertainty:
            out += f"±{self.uncertainty}"
        return out

    def __repr__(self):
        return self.__str__()


class OptimizationResult:
    def __init__(
        self,
        function: Callable,
        values: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
    ):
        self._function = function
        self._name = function.__name__
        argspecs = getfullargspec(function)
        self._arguments = argspecs.args
        self._parameters = argspecs.kwonlyargs
        if uncertainties is None:
            uncertainties = {}
        self._fit_values = {
            p: ParameterValue(values[p], uncertainties.get(p, None)) for p in values
        }

    def __getattr__(self, param):
        return self._fit_values[param]

    def __call__(self, args):
        args = np.array(args)
        if args.ndim == 1:
            args = args[None]
        return self._function(
            *np.array(args), **{k: v.value for k, v in self._fit_values.items()}
        )

    def __repr__(self):
        args = ", ".join(self._arguments)
        params = ", ".join([f"{k}={v}" for k, v in self._fit_values.items()])
        return f"<OptimizationResult {self._name}({args}; {params})>"
