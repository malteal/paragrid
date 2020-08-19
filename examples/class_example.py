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
    space_func = {'std': [1, 20, 3], 'learning_rate': [0.01, 0.1, 3], 
                   'n_estimators': [2, 50, 3]}
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target

    reg_class = test_func
    params = paragrid(model=reg_class, space=space_func,
                      X=X, y=y, target='min',
                      niter=0, func_type = 'func')
    params.gridsearch()
    param = params.score()