# Help packages
from sklearn.datasets import load_breast_cancer
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Parallel gridsearch
from paragrid import paragrid

def plot_param_space_3d(params, results):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x=np.array(params)[:,0]
    y=np.array(params)[:,1]
    z=np.array(params)[:,2]
    
    sc = ax.scatter(x,y,z, c=results)
    ax.set_xlabel('max_depth', rotation=150)
    ax.set_ylabel('learning_rate')
    ax.set_zlabel(r'n_estimators', rotation=60)
    plt.colorbar(sc)
    plt.show()

if __name__ == "__main__":
    # spaces
    space_gpdt = [Integer(2, 50, name='max_depth'),
               Real(0.01, 0.1, "log-uniform", name='learning_rate'),
               Integer(2, 50, name='n_estimators')]
    
    ncalls = 20
    
    # Classification
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target    
    reg_cls_gpdt = GradientBoostingClassifier()
    xbg_cls = XGBClassifier()
    lgbm_cls = LGBMClassifier()
    
    params = paragrid(model=reg_cls_gpdt, space=space_gpdt,
                                    X=X, y=y, ncalls = ncalls, target = 'max',
                                    niter = 1)
    param, results = params.gridsearch()
    print(params.score())
    # plot_param_space_3d(params, results)
    



