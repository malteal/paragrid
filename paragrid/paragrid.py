import inspect
import time
import types
import concurrent.futures
from tqdm import tqdm
import itertools
import numpy as np
        
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def plot_param(param, results, save: bool = False): # ! todo add 3d figure
    fig, ax = plt.subplots(figsize = (20,20))
    xlabel = np.round(list(dict.fromkeys(np.array(param)[:,0])),4)
    ylabel = np.round(list(dict.fromkeys(np.array(param)[:,1])),4)

    results = np.abs(np.array(results))
    results.shape = (len(xlabel),len(ylabel))
    # results = np.flip(results[::-1].T)
    im = plt.imshow(results)
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # labels
    ax.set_yticks(np.arange(len(xlabel))[::2])
    ax.set_xticks(np.arange(len(ylabel))[::2])
    ax.set_yticklabels(xlabel[::2], fontsize = 8)
    ax.set_xticklabels(ylabel[::2], fontsize = 8)

    plt.show()
    if save:
        plt.savefig(f'figure.png')

class paragrid():
    def __init__(self, model: types.FunctionType, space: dict, func_type: str, own_trial:bool=False, verbose=0):
        assert func_type in ['ML', 'func'], 'func_type parameter must be ML or func'
        assert isinstance(space, dict), 'The space must be a dict'
        if 'func' in func_type:
            assert isinstance(model, types.FunctionType), 'The space must be a dict'
        self.model, self.space = model, space
        self.func_type = func_type
        self.lr, self.old_parameter, self.old_results = None, None, None
        self.verbose = verbose; self.own_trial=own_trial
        if func_type == 'func':
            self.func_para = inspect.getargspec(model)[0]

        if own_trial:
            print('WARNING: own_trial has been set to True - linspace will not be used to interpolate between points.')
            
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
            
        return score, param
    
    def parallelizing(self, args, params, optimize, disable=True):
        parameter = []
        results = []
        tqdm_input = {'disable': disable, 'total': len(params)}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if not self.order:
                result = [executor.submit(self.setting_parameters, i) for i in args]
                
                for res in tqdm(concurrent.futures.as_completed(result), **tqdm_input):
                    score, param = res.result()
                    results.append(score)
                    parameter.append(param)
            else:
                results = list(tqdm(executor.map(self.setting_parameters, args), **tqdm_input))
                results,parameter = list(zip(*results))
                
        return parameter, results
    
    def create_args(self):
        self.param_names = [i for i in self.space.keys()]
        param_list = []
        if not self.own_trial:
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
        else:
            params=[i for i in itertools.product(*self.space.values())]
        # print(f'Number of iterations: {len(params)}') 
        args = ((self.model, self.param_names, b) for b in params)
        return args, params
    
    def higher_quartile(self, parameter, results):
        results = np.array(results)
        if self.target == 'min':
            # if all(results < 0):
            #     mask = results > np.percentile(results, 90)
            # elif all(results > 0):
            #     mask = results < np.percentile(results, 10)
            # else:
            mask = abs(results) < np.percentile(abs(results), 20) #@todo needs to be tested
        else: 
            mask = results > np.percentile(results, 80)

        parameter = np.array(parameter)[mask]
        parameter_low_top = []
        # for i in range(len(self.param_names)):
        #     parameter_low_top.append([np.min(parameter[mask][:,i]), np.max(parameter[mask][:,i])])
        space = []       
        if type(self.space) == dict:
            space = {}
        for name in self.param_names:
            values = [j[name] for j in parameter]
            value_types = np.array([type(i) for i in self.space[name]])
            length = len(self.space[name]) if self.own_trial else self.space[name][2]
            if all(value_types==int): # hyperparameter with int format
                self.space[name] = np.linspace(np.min(values), np.max(values), length, dtype=int)
            elif any(value_types==float): # hyperparameter with float format
                self.space[name] = np.linspace(np.min(values), np.max(values), length, dtype=float)
        args, params = self.create_args()
        return args, params
    
    def select_best(self, parameter, results):
        if self.target == 'min':
            index = np.argmin(np.abs(results))
            test = np.abs(results[index]) < np.abs(self.results_best)   
        elif self.target == 'max':
            index = np.argmax(np.abs(results))
            test = results[index] > self.results_best
            
        if test: #might giv problems, sometthing test is an 1D array, why.. dont know...
            self.parameter_best = parameter[index]
            self.results_best = results[index]


    def gridsearch(self, optimize: bool, order=True, X=None, y=None, target=None, niter=0) -> list:
        """
        Function performance the gridsearh. This can be done two main ways.

        Parameters
        ----------
        optimize : bool
            optimize=True, the algo will try to find the best parameter and return them
            optimize=False, the algo will do a simple gridsearch through the parameters given
        order : TYPE, optional
            If the return order is important for the algo. The default is True.
            order=False might be faster
        X : TYPE, optional
            If X values are needed for ML. The default is None.
            IMPORTANT: the format need to support multiple readers
        y : TYPE, optional
            If y values are needed for ML. The default is None.
            IMPORTANT: the format need to support multiple readers
        target : TYPE, optional
            Only used if optimize=True. The algo need to know to optimize or minimize. The default is None.
        niter : TYPE, optional
            How many times to rerun the gridsearc. The default is None.

        Returns
        -------
        list
            Returns a list of parameters and results.

        """
        self.optimize=optimize
        self.niter = niter; self.order = order; self.X, self.y, self.target = X, y, target
        if optimize:
            assert (target in ['min', 'max']), 'One of the optimizing parameters is not set (X, y or target)'
            self.target = target
            if self.target == 'min':
                self.results_best = 10**100
            elif self.target == 'max':
                self.results_best = 0.0
        try:
            assert any([type(i)==list for i in self.space.values()]), 'Input error: The grid space has to be list - See dokumentation'
        except SyntaxError:
            assert False, 'Input error: The grid space has to be list - See dokumentation'            
        start = time.time()

        # model_str_type = re.findall('[A-Z][^A-Z]*',str(self.model).split('(')[0])
        args, params = self.create_args()
        disable=False if self.niter==0 else True
        parameter, results = self.parallelizing(args, params, optimize, disable=disable)
        if self.optimize:
            self.select_best(parameter, results)
            for i in tqdm(range(self.niter)):
                # print('best result:', np.min(np.abs(results)))
                args, params = self.higher_quartile(parameter, results)
                parameter, results = self.parallelizing(args, params, optimize, disable=disable)
                self.select_best(parameter, results)
                # self.bool_plotting(parameter, results)
            if self.verbose:
                print(f'\nBest score: {self.results_best}')
                print(f'Parameters: {self.parameter_best}')
                print(f'Time it took: {time.time()-start}s')
                
        return parameter, results
    
    def find_gradient(self, parameter, results, number_for_mean = 10):
        # print(parameter)
        parameter_sorted = [x for _,x in sorted(zip(results, parameter))][:number_for_mean]
        mean_parameter = np.mean(parameter_sorted, axis = 0)
        self.old_space, self.old_parameter = self.space.copy(), parameter.copy()
        self.old_results = results.copy()
        for name, value in zip(self.param_names, mean_parameter):
            self.space[name] = value
        args, params = self.create_args()
        return args, params
    
    def gradient_decent(self, lr: float, valid = 20):
        self.lr = lr
        start = time.time()
        args, params = self.create_args()
        parameter, results = self.parallelizing(args, params)
        self.select_best(parameter, results)
        self.bool_plotting(parameter, results)
        jump_out_of_loop = 0 # !todo name change
        for i in tqdm(range(self.niter)):
            args, params = self.find_gradient(parameter, results)
            parameter, results = self.parallelizing(args, params)
            self.select_best(parameter, results)
            self.bool_plotting(parameter, results)
            self.plots_gradient(parameter, results, i)
            if ((self.target == 'min') & (self.results_best < np.min(results)) |
                (self.target == 'max') & (self.results_best > np.min(results))):
                jump_out_of_loop += 1
                if valid == jump_out_of_loop:
                    print('The result has not improved from'
                          f'{jump_out_of_loop} iterations')
                    break
                
        if self.verbose:
            print(f'\nBest score: {self.results_best}')
            print(f'Parameters: {self.parameter_best}')
            print(f'Time it took: {time.time()-start}s')
        return parameter, results
    
    def score(self):
        try:
            print('Best score: ', self.results_best)
            print('with parameters:', self.parameter_best)
            return self.parameter_best
        except AttributeError:
            raise AttributeError('Optimize is set to False, so the optimal parameters have not been found')
    def plots_gradient(self, parameter, result, niter):
        z = []
        size = 12
        for i, j in itertools.product(range(-size,size), range(-size,size)):
            z.append(self.model(i,j))
        parameter_sorted = [x for _,x in sorted(zip(result, parameter))]
        parameter_sorted = parameter_sorted[:10]
        t = np.array(z)
        t.shape = (2*size,2*size)
        t = np.flip(t[::-1].T)
        im = plt.imshow(t, extent=[-size, size, -size, size])
        plt.plot(np.array(parameter)[:,0], np.array(parameter)[:,1], 'b.')
        plt.plot(np.array(parameter_sorted)[:,0], np.array(parameter_sorted)[:,1], 'go')
        plt.plot(10,-7.5, 'rx')
        plt.xlabel('a')
        plt.title(f'Number of iteration: {niter}')
        plt.ylabel('b')
        plt.show()
        plt.savefig(f'./figures/{niter}.png')
        
    def bool_plotting(self, parameter, results):
        if (np.shape(parameter)[1] == 2) and self.plot:
            plot_param(parameter, results)
        elif self.plot and (np.shape(parameter)[1] != 2):
            print('Error: Can only plot 2D plots - The parameter space is not 2D')

