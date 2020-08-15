# Help packages
from sklearn.datasets import load_boston
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
import paragrid

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
    space_gpdt = {'learning_rate': [0.01, 0.1, 10],
               'n_estimators': [2, 50, 10], 'loss' : ['ls', 'lad']}
    ncalls = 100
    
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target
    reg_gpdt = GradientBoostingRegressor()
    
    params = paragrid(model=reg_gpdt, space=space_gpdt,
                                X=X, y=y, ncalls = ncalls, target = 'min',
                                niter = 0)
    
    params, results = params.gridsearch()
    
    param_best = params.score()
    # plot_param_space_3d(params, results)




