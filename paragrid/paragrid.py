import time
import warnings
import concurrent.futures
import re
from tqdm import tqdm

import numpy as np
from skopt.space import Real, Integer
from skopt import Optimizer
from sklearn.model_selection import cross_val_score

class paragrid():
    
    def __init__(self, model, space, X, y, ncalls = 10, mtype = 'res',
                 niter = 0, model_name = 'scikit'):
        self.X, self.y, self.niter = X, y, niter
        self.model, self.space = model, space
        self.ncalls, self.mtype = ncalls, mtype
        self.model_name = model_name
        
        if self.mtype == 'res':
            self.results_best = 10**10
        else:
            self.results_best = 0
    
    def objetive(self, model):
        return np.mean(cross_val_score(model, self.X, self.y, cv = 5))

    def setting_parameters(self, params):
        model, order, param = params
        param = dict(zip(order, param))
        self.model.set_params(**param)
        score = self.objetive(self.model)
        return score
    
    def parallelizing(self, args, params):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self.setting_parameters, args)
            results = []
            parameter = []
        for r, param in zip(result, params):
            results.append(r)
            parameter.append(param)
        return parameter, results

    def create_args(self):
        opt = Optimizer(self.space)
        self.param_names = [i.name for i in self.space]            
        params = [opt.ask() for _ in range(self.ncalls)]
        args = ((self.model, self.param_names, b) for b in params)
        return args, params
    
    def higher_quartile(self, parameter, results):
        if self.mtype == 'res':
            mask = results < np.percentile(results, 10)
        else: 
            mask = results > np.percentile(results, 90)

        parameter = np.array(parameter)
        parameter_low_top = []
        for i in range(len(self.param_names)):
            parameter_low_top.append([np.min(parameter[mask][:,i]), np.max(parameter[mask][:,i])])
        space = []        
        for i, j, values in zip(self.space, self.param_names, parameter_low_top):
            if i.dtype == np.int64:
                if values[0]==values[1]:
                    values = [values[0], values[1]+1]
                space.append(Integer(values[0], values[1], name=j))
            elif i.dtype == float:
                if values[0]==values[1]:
                    values = [values[0], values[1]+values[1]/100]
                space.append(Real(values[0], values[1], name=j))
        self.space = space
        args, params = self.create_args()
        return args, params
    
    def select_best(self, parameter, results):
        if self.mtype == 'res':
            index = np.argmin(np.abs(results))
            test = results[index] < self.results_best
            
        else:
            index = np.argmax(np.abs(results))
            test = results[index] > self.results_best
            
        if test:
            self.parameter_best = dict(zip(self.param_names, parameter[index]))
            self.results_best = results[index]

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
        
        args, params = self.create_args()
        parameter, results = self.parallelizing(args, params)
        self.select_best(parameter, results)
        for i in tqdm(range(self.niter)):
            print(np.max(results))
            args, params = self.higher_quartile(parameter, results)
            parameter, results = self.parallelizing(args, params)
            self.select_best(parameter, results)

        
        print(f'\nBest score: {self.results_best}')
        print(f'Parameters: {self.parameter_best}')
        print(f'Time it took: {time.time()-start}s')
        return parameter, results
    



