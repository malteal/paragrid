# Help packages
from sklearn.datasets import load_boston
import numpy as np

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
from paragrid import paragrid

if __name__ == "__main__":
    
    # spaces
    space_gpdt = {'learning_rate': [0.001, 0.1, 10],
               'n_estimators': [2, 70, 10]}
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target
    reg_gpdt = GradientBoostingRegressor()
    
    params = paragrid(model=reg_gpdt, space=space_gpdt, func_type='ML')
    
    param, results = params.gridsearch(optimize=True,X=X, y=y, target = 'min', order=True, niter=2)

    print(params.score())
    best_param = params.score()
    

