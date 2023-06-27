import numpy as np
from optimazing import OptimizationResult


def linear(x, *, a, b):
    return a * x + b


def test_optimization_result_callable():
    result = OptimizationResult(linear, {"a": 1.0, "b": 1.0}, 0.0, {"a": 0.1, "b": 0.1})
    evaluated_values = result([0, 1, 2, 3])
    assert np.allclose(evaluated_values, [1.0, 2.0, 3.0, 4.0])
    assert result.a.value == 1.0
    assert result.a.uncertainty == 0.1
    assert result.b.value == 1.0
    assert result.b.uncertainty == 0.1
    assert result._function_value == 0.0


def test_optimization_result_repr():
    result = OptimizationResult(linear, {"a": 1.0, "b": 1.0}, 0.0, {"a": 0.1, "b": 0.1})
    assert repr(result) == "<OptimizationResult linear(x; a=1.0±0.1, b=1.0±0.1)>"
