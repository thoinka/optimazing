# foptima

## Install

Clone this repository and install it with pip:

```bash
git clone https://github.com/thoinka/foptima.git
cd foptima
pip install .
```

## Usage

foptima is designed as a simple wrapper around `scipy.optimize.minimize` to
make it easier to use for common optimization tasks. The main entry point are the
`optimizable` and `loss` decorators, which take a function and returns it as an
`OptimizableFunction` or `BaseLoss` object respectively.

### `optimizable` Decorator

Here's an example of how to use `optimizable`:

```python
>>> from foptima import optimizable
...
... @optimizable
... def linear(x, *, a, b):
...     return a * x + b
...
... result = linear.fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0)
... result
 <OptimizationResult linear(x; m=1.099999957329345±0.6324555520828586, b=1.5000001036670962±1.7320508637731515)>
```

In this example, `result` will be a `foptima.FitResult`-object that contains
the result of the optimization. It can also be treated like a function, so you can
perform `result([1, 2, 3, 4])` to get the fitted values for the given x-values.
The two keyword arguments `a` and `b` in fit are the initial guesses for the
optimization.
There's also the option to use a pandas DataFrame as first argument to fit and a
column name as the second argument, like so:

```python
>>> linear.fit(df, "target")
```
The column names for the arguments in df have to match up with the argument names
in your function definition. If you can't guarantee that, then you'll have to rename
your df or your function definition, depending on what breaks your heart less.

### Freezing and Bounding Parameters

There's two more methods for `OptimizableFunction`-objects: `freeze` and `bound`. They
can be used to fix parameters to a certain value or to restrict them to a certain
range. Here's an example:

```python
>>> from foptima import optimizable
...
... @optimizable
... def linear(x, *, a, b):
...     return a * x + b
...
... result = (
...     linear.freeze(b=0.0).fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0)
... )
... result
 <OptimizationResult linear(x; m=1.0±0.0, b=0.0±0.0)>
```

or with bounds:

```python
>>> from foptima import optimizable
...
... @optimizable
... def linear(x, *, a, b):
...     return a * x + b
...
... result = (
...     linear.bound(a=(0.0, 2.0)).fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0)
... )
... result
 <OptimizationResult linear(x; m=1.0±0.0, b=0.0±0.0)>
```

### Weights and/or Uncertainties

You have both the options to pass weights and uncertainties. Depending on the specifics
of the loss you're using, they will be treated differently.

```python
linear.fit([1, 2, 3, 4], [2, 4, 6, 5], weights=[1, 0.75, 0.5, 0.5], sigma=[1.0, 2.0, 3.0, 4.0], a=1.0, b=2.0)
```

Whether combining both those inputs makes sense, of course depends on your use case.

As before, there's also the option to use a pandas DataFrame instead:

```python
linear.fit(df, target="target_col", weights="weight_col", sigma="sigma_col", a=1.0, b=2.0)
```

### `loss` Decorator

In the examples above, no loss was specified; this translates into the default loss,
which is a chi squared loss (or, if used without weights, a mean squared error loss).
You are able to select the loss function to use with the `loss`-keyword in the `fit`
method:

```python
>>> linear.fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0, loss="laplace")
```
There's a small set of pre-registered loss functions that foptima is shipped with that
you can select with a simple string like above, which can be seen when you import
`foptima.losses`:

```python
>>> from foptima import losses
>>> losses
 chi_squared: <loss chi_squared(y_true, y_est, weights, sigma)>
 laplace: <loss laplace(y_true, y_est, weights, sigma)>
 poisson: <loss poisson(y_true, y_est, weights, sigma)>
```

You can define your own losses by decorating a function with the `loss` decorator:

```python
>>> from foptima import loss
>>> @loss
... def mse(y_true, y_est, weights, sigma):
...     return np.mean((y_true - y_est) ** 2)
```
Now the object `mse` can be used as a loss function:

```python
>>> linear.fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0, loss=mse)
```

Alternatively, you can also register the loss function with losses, so that they can
be used like the built-in ones:

```python
>>> from foptima import loss
>>> @loss(register=True)
... def mse(y_true, y_est, weights, sigma):
...     return np.mean((y_true - y_est) ** 2)
>>> linear.fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0, loss="mse")
```

Additionally, it is possible to pass parameters to a loss function:

```python
>>> @loss
... def student_t(y_true, y_est, weights, sigma, *, nu=1.0)
...     diff = (y_true - y_est) / sigma
...     return np.mean(weights * np.log(1 + diff ** 2 / nu) * ((nu + 1) / 2))
>>> linear.fit([1, 2, 3, 4], [2, 4, 6, 5], a=1.0, b=2.0, loss=student_t(nu=4.0))
```

Note that you have to declare it as a keyword-only argument. Also, note that
it is currently not possible to make such parameters optimizable.
## Caveats

Being a wrapper around `scipy.optimize.minimize`, foptima introduces quite a bit of
overhead. If you're optimizing a function that is very fast to evaluate, you
might be better off using `scipy.optimize.minimize` directly. Without any promises,
foptima will run something like 10% slower than `scipy.optimize.minimize`. Whether this
is relevant for your use case is up to you to decide.