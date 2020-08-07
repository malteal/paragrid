# Help packages
from sklearn.datasets import load_boston
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor

# Parallel gridsearch
from paragrid import paragrid

def test_func(std, mean):
    reg_gpdt = GradientBoostingRegressor(loss='lad')
    score = 1
    return score

if __name__ == "__main__":
    # spaces
    space_gpdt = [Integer(2, 50, name='max_depth'),
                  Real(0.01, 0.1, "log-uniform", name='learning_rate'),
                  Integer(2, 50, name='n_estimators')
                  ]
    ncalls = 10

    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target

    reg_class = test_func(std = 1, mean = 0)

    params = paragrid(model=reg_class, space=space_gpdt,
                      X=X, y=y, ncalls=ncalls, mtype='res',
                      niter=0)

