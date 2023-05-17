import numpy as np
from foptima import OptimizableFunction
import pytest
from scipy.optimize import minimize


@pytest.fixture
def data():
    x = np.linspace(0.0, 1.0, 16)
    y = 3 * x + 1
    y += np.random.RandomState(0).randn(16) * 0.1
    return x, y


def linear(x, *, a, b):
    return a * x + b


@pytest.mark.parametrize("a0,b0", [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]])
def test_regular_fit(data, a0, b0):
    x, y = data
    optfun = OptimizableFunction(linear)
    result = optfun.fit(x, y, a=a0, b=b0)
    exp = minimize(lambda p: np.mean((y - p[0] * x - p[1]) ** 2), x0=[a0, b0])
    np.testing.assert_almost_equal(result.a.value, exp.x[0], decimal=7)
    np.testing.assert_almost_equal(result.b.value, exp.x[1], decimal=7)


@pytest.mark.parametrize(
    "a_bound,b_bound",
    [
        [(0.0, 10.0), (0.0, 2.0)],
        [(1.0, 2.0), (0.0, 2.0)],
        [(-1.0, 10.0), (-1.0, 0.0)],
        [(1.0, 2.0), (None, None)],
        [(None, None), (-1.0, 0.0)],
    ],
)
def test_bounded_fit(data, a_bound, b_bound):
    x, y = data
    optfun = OptimizableFunction(linear)
    bounds = {}
    if a_bound != (None, None):
        bounds.update(a=a_bound)
    if b_bound != (None, None):
        bounds.update(b=b_bound)
    result = optfun.bound(**bounds).fit(x, y)
    exp = minimize(
        lambda p: np.mean((y - p[0] * x - p[1]) ** 2),
        x0=[1.0, 1.0],
        bounds=[a_bound, b_bound],
    )
    np.testing.assert_almost_equal(result.a.value, exp.x[0], decimal=7)
    np.testing.assert_almost_equal(result.b.value, exp.x[1], decimal=7)


@pytest.mark.parametrize(
    "a_freeze,b_freeze",
    [
        [3.0, None],
        [None, 1.0],
        [1.0, None],
        [None, -1.0],
    ],
)
def test_frozen_fit(data, a_freeze, b_freeze):
    x, y = data
    optfun = OptimizableFunction(linear)
    frozen = {}
    if a_freeze is not None:
        frozen.update(a=a_freeze)
    if b_freeze is not None:
        frozen.update(b=b_freeze)
    result = optfun.freeze(**frozen).fit(x, y)
    if a_freeze is None:

        def func(p):
            return np.mean((y - p * x - b_freeze) ** 2)

    elif b_freeze is None:

        def func(p):
            return np.mean((y - a_freeze * x - p) ** 2)

    exp = minimize(func, x0=[1.0])
    if a_freeze is None:
        np.testing.assert_almost_equal(result.a.value, exp.x[0], decimal=7)
    else:
        np.testing.assert_almost_equal(result.b.value, exp.x[0], decimal=7)
