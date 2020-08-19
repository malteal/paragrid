import inspect
import time
import warnings
import concurrent.futures
import re
from tqdm import tqdm
import math
import itertools

import numpy as np
import skopt.space
from skopt.space import Real, Integer
from skopt import Optimizer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def plot_param(param, results):
    param = np.array(param)
    results = np.array(results)
    results.shape = (len(results), )
    all_params = np.zeros((len(results),3))
    all_params[:, :2] = param
    all_params[:, 2] = results
    plt.scatter(all_params[:,0], all_params[:,1],
        s = np.abs(all_params[:,2])*100, color = 'red')
    
    best = all_params[np.argpartition(np.abs(all_params[:,2]), 4)][:5]
    plt.scatter(best[:,0], best[:,1],
        s = np.abs(best[:,2])*100, color = 'blue')
    plt.savefig(f'{np.mean(np.abs(best[:,2]))}.png')

class paragrid():
    
    def __init__(self, model, space: [list, dict], X=None, y=None, ncalls = 10, target = 'min',
                 niter = 0, func_type = 'ML', plot = False):
        assert target in ['min', 'max'], 'target parameter must be min or max'
        assert func_type in ['ML', 'func'], 'func_type parameter must be ML or func'
        
        self.X, self.y, self.niter = X, y, niter
        self.model, self.space = model, space
        self.ncalls, self.target = ncalls, target
        self.func_type = func_type
        self.plot = plot
        if func_type == 'func':
            self.func_para = inspect.getargspec(model)[0]
        
        if self.target == 'min':
            self.results_best = 10**10
        elif self.target == 'max':
            self.results_best = 0.0
            
        if type(space) == dict:
            # warnings.warn("ncalls not being used")
            print('Warning: ncalls not being used')
            
    def objetive(self, model):
        return np.mean(cross_val_score(model, self.X, self.y, cv = 5))

    def setting_parameters(self, params):
        model, order, param = params
        param = dict(zip(order, param))
        
        if self.func_type == 'ML':
            self.model.set_params(**param)
            score = self.objetive(self.model)
        elif self.func_type == 'func':
            if ('X' in self.func_para):
                param['X'] = self.X
            if ('y' in self.func_para):
                param['y'] = self.y
                
            assert ~any([i not in param.keys() for i in self.func_para]), 'Parameter missing'
            score = self.model(**param)
            
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
        if type(self.space) == dict:
            self.param_names = [i for i in self.space.keys()]
            param_list = []
            for i in self.param_names:
                dtype = ('int' if all([type(i) == int for i in self.space[i][:2]])
                         else 'float' if all([type(i) == float for i in self.space[i][:2]])
                         else 'str')
                if dtype == 'str':
                    param_list.append(np.array(self.space[i]))
                else:
                    param_list.append(np.linspace(self.space[i][0],
                                                  self.space[i][1],
                                                  self.space[i][2],
                                                  dtype = dtype))
            params = [i for i in itertools.product(*param_list)]
            print(f'Number of iterations: {len(params)}')
        else:
            opt = Optimizer(self.space)
            self.param_names = [i.name for i in self.space]            
            params = [opt.ask() for _ in range(self.ncalls)]
            
        args = ((self.model, self.param_names, b) for b in params)
        return args, params
    
    def higher_quartile(self, parameter, results):
        results = np.array(results)
        if self.target == 'min':
            if all(results < 0):
                mask = results > np.percentile(results, 90)
            elif all(results > 0):
                mask = results < np.percentile(results, 10)
            else:
                mask = abs(results) < np.percentile(abs(results), 30)
        else: 
            mask = results > np.percentile(results, 90)

        parameter = np.array(parameter)
        parameter_low_top = []
        for i in range(len(self.param_names)):
            parameter_low_top.append([np.min(parameter[mask][:,i]), np.max(parameter[mask][:,i])])
        space = []       
        if type(self.space) == dict:
            space = {}
        for i, j, values in zip(self.space, self.param_names, parameter_low_top):
            try: # for skopt type
                if i.dtype == int:
                    if values[0]==values[1]:
                        values = [values[0], values[1]+1]
                    space.append(Integer(values[0], values[1], name=j))
                elif i.dtype == float:
                    if values[0]==values[1]:
                        values = [values[0], values[1]+values[1]/100]
                    space.append(Real(values[0], values[1], name=j))
            except AttributeError as e: # for dict type
                if (type(self.space[i][0]) == int) & (type(self.space[i][1]) == int):
                    if type(values[0])==type(values[1]):
                        values = [int(values[0]-values[1]/100), 
                                  int(values[1]+values[1]/100),self.space[i][2]]
                    space[i] = values
                elif (type(self.space[i][0]) == float) & (type(self.space[i][1]) == float):
                    if type(values[0])==type(values[1]):
                        values = [values[0]-values[1]/100, 
                                  values[1]+values[1]/100,self.space[i][2]]
                    space[i] = values
        self.space = space
        args, params = self.create_args()
        return args, params
    
    def select_best(self, parameter, results):
        if self.target == 'min':
            index = np.argmin(np.abs(results))
            test = results[index] < self.results_best   
        elif self.target == 'max':
            index = np.argmax(np.abs(results))
            test = results[index] > self.results_best
            
        try:
            if any(test): # sometthing test is an 1D array, why.. dont know...
                self.parameter_best = dict(zip(self.param_names, parameter[index]))
                self.results_best = results[index]
        except:
            if test:
                self.parameter_best = dict(zip(self.param_names, parameter[index]))
                self.results_best = results[index]

    def gridsearch(self):
        start = time.time()

        # model_str_type = re.findall('[A-Z][^A-Z]*',str(self.model).split('(')[0])
        args, params = self.create_args()
        parameter, results = self.parallelizing(args, params)
        self.select_best(parameter, results)
        self.bool_plotting(parameter, results)
        for i in tqdm(range(self.niter)):
            args, params = self.higher_quartile(parameter, results)
            parameter, results = self.parallelizing(args, params)
            self.select_best(parameter, results)
            self.bool_plotting(self, parameter, results)
        
        print(f'\nBest score: {self.results_best}')
        print(f'Parameters: {self.parameter_best}')
        print(f'Time it took: {time.time()-start}s')
        return parameter, results
    

    def score(self):
        return self.parameter_best
    

    def bool_plotting(self, parameter, results):
        if (np.shape(parameter)[1] == 2) and self.plot:
            plot_param(parameter, results)
        elif self.plot and (np.shape(parameter)[1] != 2):
            print('Error: Can only plot 2D plots - The parameter space is not 2D')

