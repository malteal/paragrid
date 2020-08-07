# Help packages
from sklearn.datasets import load_boston
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor

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
               Integer(2, 50, name='n_estimators'),
               # Real(10**-3, 1, name='subsample'),
              # Integer(2, X.shape[1], name='max_features'),
              ]
    ncalls = 10
    
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target
    reg_gpdt = GradientBoostingRegressor(loss = 'lad')
    
    params = paragrid(model=reg_gpdt, space=space_gpdt,
                                X=X, y=y, ncalls = ncalls, mtype = 'res',
                                niter = 0)
    
    params, results = params.gridsearch()
    
    plot_param_space_3d(params, results)




