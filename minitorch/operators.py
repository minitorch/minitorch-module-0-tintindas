"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: int | float, b: int | float) -> int | float:
    """Multiplies two int | floats."""
    return a * b


# - id
def id(a: int | float) -> int | float:
    """Returns input unchanged."""
    return a


# - add
def add(a: int | float, b: int | float) -> int | float:
    """Adds two int | floats."""
    return a + b


# - neg
def neg(a: int | float) -> int | float:
    """Negates input."""
    return -a


# - lt
def lt(a: int | float, b: int | float) -> bool:
    """Checks if one int | float is less than the other."""
    return a < b


# - eq
def eq(a: int | float, b: int | float) -> bool:
    """Checks if two int | floats are equal."""
    return a == b


# - max
def max(a: int | float, b: int | float) -> int | float:
    """Returns the larger of two int | floats."""
    return a if a > b else b


# - is_close
def is_close(a: int | float, b: int | float) -> bool:
    """Checks if two int | floats are close in value."""
    return abs(a - b) < 1e-2


# - sigmoid
def sigmoid(x: int | float) -> int | float:
    """Applies sigmoid function to input."""
    return (1 / (1 + math.exp(-x))) if x >= 0 else (math.exp(x) / (1 + math.exp(x)))


# - relu
def relu(x: int | float) -> int | float:
    """Applies RelU activation function to input."""
    return x if x >= 0 else 0


# - log
def log(x: int | float) -> float:
    """Calculates the natural logarithm of the input."""
    return math.log(x)


# - exp
def exp(x: int | float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


# - log_back
def log_back(x: int | float, grad_other: int | float) -> float:
    """Calculates the derivative of the natural logarithm times a second argument."""
    return (1 / x) * grad_other


# - inv
def inv(x: int | float) -> float:
    """Calculates the reciprocal."""
    return 1 / x


# - inv_back
def inv_back(x: int | float, grad_other: int | float) -> float:
    """Calculates the derivative of the inverse times a second argument."""
    return (-1 / x**2) * grad_other


# - relu_back
def relu_back(x: int | float, grad_other: int | float) -> float:
    """Calculates the derivative of RelU times a second argument."""
    return grad_other if x >= 0 else 0.0


# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable, arr: Iterable) -> Iterable:
    """Applies given function to every element of given array."""
    return [fn(item) for item in arr]


# - zipWith
def zipWith(fn: Callable, arr1: Iterable, arr2: Iterable) -> Iterable:
    """Applies given function to every pair (index-wise) of given arrays."""
    res = []
    for a, b in zip(arr1, arr2):
        res.append(fn(a, b))
    return res


# - reduce
def reduce(fn: Callable, arr: Iterable, start: int | float) -> int | float | bool:
    """Applies given function which reduces the given array to a single value."""
    res = start
    for item in arr:
        res = fn(res, item)
    return res


# Use these to implement
# - negList : negate a list
def negList(arr: Iterable) -> Iterable:
    """Negates a list."""
    return map(lambda x: -x, arr)


# - addLists : add two lists together
def addLists(arr1: Iterable, arr2: Iterable) -> Iterable:
    """Add two lists element-wise."""
    return zipWith(add, arr1, arr2)


# - sum: sum lists
def sum(arr: Iterable) -> int | float:
    """Sum all elements of a list."""
    return reduce(add, arr, 0)


# - prod: take the product of lists
def prod(arr: Iterable) -> int | float:
    """Calculate the product of all the elements of a list."""
    return reduce(mul, arr, 1)
