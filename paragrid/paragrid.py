import inspect
import time
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
    def __init__(self, model: [], space: dict, X=None,  y=None, target = 'min',
                 niter = 0, func_type = 'ML', plot = False):
        assert target in ['min', 'max'], 'target parameter must be min or max'
        assert func_type in ['ML', 'func'], 'func_type parameter must be ML or func'
        plt.close('all')
        self.X, self.y, self.niter = X, y, niter
        self.model, self.space = model, space
        self.target = target
        self.func_type = func_type
        self.plot = plot
        self.lr, self.old_parameter, self.old_results = None, None, None
        
        if func_type == 'func':
            self.func_para = inspect.getargspec(model)[0]
        
        if self.target == 'min':
            self.results_best = 10**100
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
            if self.niter == 0:
                results = list(tqdm(executor.map(self.setting_parameters, args),
                                    total = len(params)))
            else:
                results = list(executor.map(self.setting_parameters, args))
            parameter = []

        for param in params:
            parameter.append(param)
        return parameter, results
    
    def create_args(self):
        self.param_names = [i for i in self.space.keys()]
        param_list = []
        if self.lr is None:
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
            self.lr = 1/self.lr #!!!todo to be changed reducing lr for the radius
            N = 100 #@todo this nees to be an attr
            dim = len(self.param_names)
            norm = np.random.normal
            normal_deviates = norm(size=(dim, N))
            radius = np.sqrt((normal_deviates**2).sum(axis=0))*self.lr
            points = np.array(normal_deviates/radius)
            parameter = np.array(list(self.space.values()))
            parameter.shape = (dim,1)
            params = parameter+points
            if (self.old_parameter is not None) & (self.old_results is not None):
                old_center = np.array(list(self.old_space.values()))
                old_center.shape = (dim,1)
                params_old = old_center-params
                # print(np.sqrt(np.sum(old_center.T-np.array(list(self.parameter_best.values())))**2))
                params = np.array(tuple(map(tuple, params.T)))[np.sqrt(np.sum(params_old**2,axis=0))>1]
                params = tuple(map(tuple, params))
            else:
                params = tuple(map(tuple, params.T))
        # print(f'Number of iterations: {len(params)}') 
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
            self.bool_plotting(parameter, results)
        
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
                
        
        print(f'\nBest score: {self.results_best}')
        print(f'Parameters: {self.parameter_best}')
        print(f'Time it took: {time.time()-start}s')
        return parameter, results
    
    def score(self):
        return self.parameter_best
    
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

