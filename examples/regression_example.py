# Help packages
from sklearn.datasets import load_boston
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
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
    space_gpdt = {'learning_rate': [0.001, 0.1, 10],
               'n_estimators': [2, 70, 10],
               #'loss' : ['ls', 'lad']}
               }
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target
    reg_gpdt = GradientBoostingRegressor()
    
    params = paragrid(model=reg_gpdt, space=space_gpdt,
                                X=X, y=y, target = 'min',
                                niter = 1)
    
    param, results = params.gridsearch()
    
    # plot_param_space_3d(params, results)

    #%%
    # param = np.array(param)
    # results = np.array(results)
    # results.shape = (len(results), )
    # all_params = np.zeros((len(results),3))
    # all_params[:, :2] = param
    # all_params[:, 2] = results
    # plt.scatter(all_params[:,0], all_params[:,1],
    #             s = np.abs(all_params[:,2])*100, color = 'red')

    # best = all_params[np.argpartition(np.abs(all_params[:,2]), 4)][:5]
    # plt.scatter(best[:,0], best[:,1],
    #             s = np.abs(best[:,2])*10000, color = 'blue')



