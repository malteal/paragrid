# Help packages
from sklearn.datasets import load_boston
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Parallel gridsearch
from paragrid import paragrid

def test_func(X, y, std, learning_rate, n_estimators):
    mask = std<np.std(X, axis = 0)
    X = X[:,mask] ## setting restriction on std of columns
    reg_gpdt = GradientBoostingRegressor(loss = 'lad',
                                         learning_rate = learning_rate,
                                         n_estimators = n_estimators)
    return np.mean(cross_val_score(reg_gpdt, X, y, cv = 5))

if __name__ == "__main__":
    # spaces
    # space_func = [Real(1, 20, "log-uniform", name='std'),
    #               Real(0.01, 0.1, "log-uniform", name='learning_rate'),
    #               Integer(2, 50, name='n_estimators')]
    space_func = [{'std': [1, 20], 'learning_rate': [0.01, 0.1],
                   'n_estimators': [2, 50]}]
    ncalls = 10

    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target

    reg_class = test_func
    params = paragrid(model=reg_class, space=space_func,
                      X=X, y=y, ncalls=ncalls, target='min',
                      niter=0, func_type = 'func')
    params.gridsearch()
    param = params.score()