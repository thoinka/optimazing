# foptima

foptima is designed as a simple wrapper around `scipy.optimize.minimize` to
make it easier to use for common optimization tasks. The main entry point is the `optimizable`-decorator, which takes a function and returns it as an `OptimizableFunction`-object. All keyword-only arguments will be treated as optimizable parameters. Here's a very basic example:

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