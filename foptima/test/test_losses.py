import numpy as np
from foptima import losses
import pytest


@pytest.mark.parametrize("loss", list(losses._losses.values()))
def test_losses(loss):
    rnd = np.random.RandomState(0)
    evaluated_loss = loss(rnd.rand(10), rnd.rand(10), rnd.rand(10))
    assert isinstance(evaluated_loss, float)
