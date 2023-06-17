from foptima import optimizable, OptimizableFunction
import pytest


def test_optimizable():
    @optimizable
    def linear(x, *, a, b):
        return a * x + b

    assert isinstance(linear, OptimizableFunction)
    assert linear._arguments == ["x"]
    assert linear._parameters == ["a", "b"]
    assert linear._freeze_dict == {}
    assert linear._bounds_dict == {}


def test_optimizable_raise_when_no_params():
    with pytest.raises(SyntaxError):

        @optimizable
        def linear(x, a, b):
            return a * x + b


def test_optimizable_raise_when_no_args():
    with pytest.raises(SyntaxError):

        @optimizable
        def linear(*, x, a, b):
            return a * x + b
