# Help packages
from sklearn.datasets import load_breast_cancer
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
    space_gpdt = {'learning_rate': [0.001, 0.1, 5],
                  'n_estimators': [2, 70, 5],
                  'max_depth': [2, 50, 4]}

    
    # Classification
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target    
    reg_cls_gpdt = GradientBoostingClassifier()
    xbg_cls = XGBClassifier()
    lgbm_cls = LGBMClassifier()
    
    params = paragrid(model=reg_cls_gpdt, space=space_gpdt,
                                    X=X, y=y, target = 'max',
                                    niter = 1)
    param, results = params.gridsearch()
    print(params.score())
    



