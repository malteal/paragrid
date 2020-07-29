import time
import warnings
import concurrent.futures
import re
from tqdm import tqdm

import numpy as np


from skopt.utils import use_named_args
from skopt import gp_minimize, Optimizer

from sklearn.model_selection import cross_val_score

# https://github.com/cgnorthcutt/hypopt

#%%
class hyper_parameter_tuning():
    
    def __init__(self, model, space, X, y, ncalls = 10, mtype = 'res'):
        self.X, self.y = X, y
        self.model, self.space = model, space
        self.ncalls, self.mtype = ncalls, mtype
    
    
    def objetive(self, model):
        return np.mean(cross_val_score(model, self.X, self.y, cv = 5))

    def setting_parameters(self, params):
        model, order, param = params
        param = dict(zip(order, param))
        self.model.set_params(**param)
        score = self.objetive(self.model)
        return score

    def gridsearch(self):
        start = time.time()
        ## warnings/errors
        mtypes = ['res', 'cls']
        if self.mtype not in mtypes:
            raise ValueError("Invalid model type. Expected one of: %s" % mtypes)
            
        model_str_type = re.findall('[A-Z][^A-Z]*',str(self.model).split('(')[0])
        if (('Regressor' in model_str_type and self.mtype == 'cls') or 
            ('Classifier' in model_str_type and self.mtype == 'res')):
            warnings.warn("model type and model is not the same - optimizer might not be correct")
        
        ## parallelizing
        param_names = []
        for i in self.space:
            param_names.append(i.name)
            
        opt = Optimizer(self.space)
        params = []
        for i in range(self.ncalls):
            params.append(opt.ask())
        args = ((self.model, param_names, b) for b in params)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self.setting_parameters, args)
            results = []
            parameter = []
            for r, param in tqdm(zip(result, params)):
                results.append(r)
                parameter.append(param)
        if self.mtype == 'res':
            index = np.argmin(np.abs(results))
        else:
            index = np.argmax(np.abs(results))
        parameter = dict(zip(param_names, parameter[index]))
        print(f'\nMinimum score: {results[index]}')
        print(f'Parameters: {parameter}')
        print(f'Time it took: {time.time()-start}s')
        return parameter
    



