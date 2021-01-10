# Help packages
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier


# Parallel gridsearch
from paragrid import paragrid


if __name__ == "__main__":
    # spaces
    space_gpdt = {'learning_rate': [0.1,0.3,0.4, 0.6,0.8, 1],
                  'n_estimators': [200, 400, 600, 800, 1000],
                  'max_depth': [2]}

    
    # Classification
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target    
    reg_cls_gpdt = GradientBoostingClassifier()
    # xbg_cls = XGBClassifier()
    lgbm_cls = LGBMClassifier()
    
    params = paragrid(model=reg_cls_gpdt, space=space_gpdt, func_type='ML', own_trial=True)
    
    param, results = params.gridsearch(optimize=True,X=X, y=y, target = 'max', order=False, niter=5)

    print(params.score())
    best_param = params.score()
    



