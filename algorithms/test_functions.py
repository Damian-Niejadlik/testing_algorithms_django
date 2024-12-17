import numpy as np
from typing import Iterable


# Parametry dla funkcji optymalizacyjnych:
max_iterations = [50, 100, 500, 1000]
population_sizes = [30, 50, 100]
trials = 10


# Rastrigin Function
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def rastrigin_properties():
    return {
        'name': "Rastrigin",
        'minimum': 0.0,
        'dimensions': [2, 5, 10, 30],
        'lower_boundary': -5.12,
        'upper_boundary': 5.12
    }


# Rosenbrock Function
def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def rosenbrock_properties():
    return {
        'name': "Rosenbrock",
        'minimum': 0.0,
        'dimensions': [2, 5, 10, 30],
        'lower_boundary': -5.0,
        'upper_boundary': 5.0
    }


# Sphere Function
def sphere(x):
    return np.sum(x ** 2)

def sphere_properties():
    return {
        'name': "Sphere",
        'minimum': 0.0,
        'dimensions': [2, 5, 10, 30, 50],
        'lower_boundary': -5.12,
        'upper_boundary': 5.12
    }


# Beale Function
def beale(x):
    x1, x2 = x[0], x[1]
    term1 = (1.5 - x1 + x1 * x2) ** 2
    term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
    term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
    return term1 + term2 + term3

def beale_properties():
    return {
        'name': "Beale",
        'minimum': 0.0,
        'dimensions': [2],
        'lower_boundary': -4.5,
        'upper_boundary': 4.5
    }


# Bukin Function
def bukin(x):
    x1, x2 = x[0], x[1]
    x1 = max(min(x1, -5.0), -15.0)
    x2 = max(min(x2, 3.0), -3.0)
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
    term2 = 0.01 * np.abs(x1 + 10)
    return term1 + term2

def bukin_properties():
    return {
        'name': "Bukin",
        'minimum': 0.0,
        'dimensions': [2],
        'lower_boundary': -15.0,
        'upper_boundary': 3.0
    }


# Himmelblaus Function
def himmelblaus(x):
    x1, x2 = x[0], x[1]
    term1 = (x1 ** 2 + x2 - 11) ** 2
    term2 = (x1 + x2 ** 2 - 7) ** 2
    return term1 + term2

def himmelblaus_properties():
    return {
        'name': "Himmelblaus",
        'minimum': 0.0,
        'dimensions': [2],
        'lower_boundary': -5.0,
        'upper_boundary': 5.0
    }
