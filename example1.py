from sklearn.datasets import load_boston, load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer

# Classifiers
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from functions import hyper_parameter_tuning
boston = load_boston()
breast_cancer = load_breast_cancer()
X, y = boston.data, boston.target
X, y = breast_cancer.data, breast_cancer.target

# Classifiers
reg_gpdt = GradientBoostingRegressor(loss = 'lad')
reg_knn = KNeighborsRegressor()
reg_cls_gpdt = GradientBoostingClassifier()

# spaces
space_gpdt = [Integer(1, 20, name='max_depth'),
          Real(10**-3, 10**-1, "log-uniform", name='learning_rate'),
          Integer(2, X.shape[1], name='max_features'),
          Integer(3, 50, name='n_estimators')]

space_knn = [Integer(3, 400, name='n_neighbors')]


params = hyper_parameter_tuning(model=reg_cls_gpdt, space=space_gpdt,
                                X=X, y=y, ncalls = 100, mtype = 'cls')
params.gridsearch()


#%% 
import time
import warnings
import concurrent.futures
import re
from tqdm import tqdm

import numpy as np

from skopt.utils import use_named_args
from skopt import gp_minimize, Optimizer

from sklearn.model_selection import cross_val_score


def objetive(model):
    return np.mean(cross_val_score(model, X, y, cv = 5))

def setting_parameters(params):
    model, order, param = params
    param = dict(zip(order, param))
    model.set_params(**param)
    score = objetive(model)
    return score

def gridsearch(model, space, X, y, ncalls, mtype):
    start = time.time()
    ## warnings/errors
    mtypes = ['res', 'cls']
    if mtype not in mtypes:
        raise ValueError("Invalid model type. Expected one of: %s" % mtypes)
        
    model_str_type = re.findall('[A-Z][^A-Z]*',str(model).split('(')[0])
    if (('Regressor' in model_str_type and mtype == 'cls') or 
        ('Classifier' in model_str_type and mtype == 'res')):
        warnings.warn("model type and model is not the same - optimizer might not be correct")
    
    ## parallelizing
    param_names = []
    for i in space:
        param_names.append(i.name)
        
    opt = Optimizer(space)
    params = []
    for i in range(ncalls):
        params.append(opt.ask())
    args = ((model, param_names, b) for b in params)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(setting_parameters, args)
        results = []
        parameter = []
        for r, param in tqdm(zip(result, params)):
            results.append(r)
            parameter.append(param)
    if mtype == 'res':
        index = np.argmin(np.abs(results))
    else:
        index = np.argmax(np.abs(results))
    parameter = dict(zip(param_names, parameter[index]))
    print(f'\nMinimum score: {results[index]}')
    print(f'Parameters: {parameter}')
    print(f'Time it took: {time.time()-start}s')
    return parameter
    

params = gridsearch(model=reg_cls_gpdt, space=space_gpdt,
                                X=X, y=y, ncalls = 100, mtype = 'cls')


