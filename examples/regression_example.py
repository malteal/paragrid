# Help packages
from sklearn.datasets import load_boston

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
from paragrid import paragrid

if __name__ == "__main__":
    
    # spaces
    space_gpdt = {'learning_rate': [0.001, 0.1, 20],
               'n_estimators': [2, 70, 20]}
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target
    reg_gpdt = GradientBoostingRegressor()
    
    params = paragrid(model=reg_gpdt, space=space_gpdt,
                                X=X, y=y, target = 'min',
                                niter = 0)
    
    param, results = params.gridsearch()
    

