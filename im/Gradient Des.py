# Frequently when doing data science, we'll be trying to find the best model for a certain situation. Usually the best
# will mean something like "minimizes the error of its predictions" or "maximizes the likelihood of the data".
# in otherwords, it will represent the solution to some sort of optimization problem.

# This means we'll need to solve a number of optimization problems. In particular, we'll need to solve them from scratch
# Our approach will be a technique called GRADIENT DESCENT, which lends itself well to a from scratch treatment

# Suppose we have some function F that takes as input a vector of real numbers and outputs a single real number
# One simple function isL

from scratch.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)

# We'll frequently need to maximize or minimize fucntions. That is, we need to find the input V that produces the
# largest (or smallest) possible value.


# For functions like ours, the GRADIENT (the vector of partial derivatives) gives the input direction in which the
# function most quickly increases.

# Accordingly, one approach to maximizing a function is to pick a random starting point, compute the gradient, take
# a small step in the direction of the gradient (i.e. the direction that causes the function to increase the most
# and repeat with the new starting point. Similarly, you can try minimize a fucntion by doing the opposite.

#-----------------------------
# Estimating the Gradient
#-----------------------------

# if F is a function of one variable, its derivitive at a point x measures how f(x) changes when we make a very small
# change to x. The derivative is defined as the limit of the difference quotients:

from typing import Callable

def diff_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x+h) - f(x)) / h
# as h approaches 0

# For many functions, it's easy to exactly calculate derivatives. For example, the SQUARE fucntion:

def square(x: float) -> float:
    return x * x

# Has the derivative:

def derivative(x:float) -> float:
    return 2 * x

# this is easy for us to check by computing th difference quotient and taking the limit
# but what if we couldnt find the gradient? Although we can't take limits in Python, we can estimate derivatives by
# evaluating the difference quotient for a very small e.

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [diff_quotient(square, x, h=0.001) for x in xs]

# plot to show they're basically the same
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs Estimates")
plt.plot(xs, actuals, "rx", label="Actual")         # red x
plt.plot(xs, estimates, "b+", label="Estimates")    # blue +
plt.legend(loc=9)
plt.show()