# Paragrid

Paragrid is a simple parallized gridsearch, that are able to utilize all cpu core.
The main focus of this package is to reduce the lines of code, one has to write to find a good estimate for the parameters for a function.
This package works for most machine learning method as well as functions (see function section).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Installing

Recent release:
```
pip install paragrid
```

To install the git codebase to add modifications:
```
git clone https://github.com/malteal/paragrid.git
```
## Usage
### Gridsearch
#### Function
Example for using paragrid to find the optimal parameters of a function.
Ex: std: from 1 to 20 and 5 points in between.
```python
from paragrid import paragrid
import numpy as np 
import matplotlib.pyplot as plt
def test_func(a,b): # some function to minimize
    if ((2.5 < a < 10.5) & (-5 > b > -6.5) | (0.5 < a < 3) & (-3.5 > b > -4.5)
        | (-3.5 < a < 0.5) & (-3.5 > b > -4.5)):
        constant = 0
    else:
        constant = -100

    return a**2+1.5*(b+2)**2+100+constant

def find_gradient(parameter, results, number_for_mean = 10):
    parameter_sorted = [x for _,x in sorted(zip(results, parameter))]
    return parameter_sorted[:number_for_mean]

if __name__ == "__main__":
    # spaces
    n = 300
    space_func = {'a':  [-10,10, n], 'b': [-10,10, n]}

    params = paragrid(model=test_func, space=space_func, func_type = 'func')
    # parameter, results = params.gradient_decent(lr = 1)
    parameters, results = params.gridsearch(optimize=False, order = True)
    # param = params.score()
    results =np.array(results)
    results.shape = (n,n)
    plt.contourf(np.linspace(-10, 10, n), np.linspace(-10, 10, n), np.array(results.T))
```
#### Classification
Example for using paragrid for classification in ML using scikit-optimize
```python
# Help packages
from sklearn.datasets import load_breast_cancer

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier


# Parallel gridsearch
from paragrid import paragrid


if __name__ == "__main__":
    # spaces
    space_gpdt = {'learning_rate': [0.1,0.3,0.4, 0.6,0.8, 1],
                  'n_estimators': [200, 400, 600, 800, 1000],
                  'max_depth': [2]}

    
    # Classification
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target    
    reg_cls_gpdt = GradientBoostingClassifier()
    # xbg_cls = XGBClassifier()
    lgbm_cls = LGBMClassifier()
    
    params = paragrid(model=reg_cls_gpdt, space=space_gpdt, func_type='ML', own_trial=True)
    
    param, results = params.gridsearch(optimize=True,X=X, y=y, target = 'max', order=False, niter=5)

    print(params.score())
    best_param = params.score()
```
#### Regression
Example for using paragrid for regression in ML with normal gridsearch.
Ex: learning_rate: from 0.01 to 0.1 and 10 points in between.
``` python
# Help packages
from sklearn.datasets import load_boston

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
from paragrid import paragrid

if __name__ == "__main__":
    
    # spaces
    space_gpdt = {'learning_rate': [0.001, 0.1, 10],
               'n_estimators': [2, 70, 10]}
    # Regression
    boston = load_boston()
    X, y = boston.data, boston.target
    reg_gpdt = GradientBoostingRegressor()
    
    params = paragrid(model=reg_gpdt, space=space_gpdt, func_type='ML')
    
    param, results = params.gridsearch(optimize=True,X=X, y=y, target = 'min', order=True, niter=2)

    print(params.score())
    best_param = params.score()
   
```
### Bruce force gradient descent
##### Not working at the moment
```diff
--Gradient descent is only working with floats as parameters. (int and/or string will not work)--
```
#### Function
Here is an examples of using the gradient descent algorithm in paragrid with the use of a test function.
    
```python
# Parallel gridsearch
from paragrid import paragrid

def test_func(a,b): # some function to minimize
    if ((2.5 < a < 10.5) & (-5 > b > -6.5) | (0.5 < a < 3) & (-3.5 > b > -4.5)
        | (-3.5 < a < 0.5) & (-3.5 > b > -4.5)):
        constant = 0
    else:
        constant = -100

    return a**2+1.5*(b+2)**2+100+constant

def find_gradient(parameter, results, number_for_mean = 10):
    parameter_sorted = [x for _,x in sorted(zip(results, parameter))]
    return parameter_sorted[:number_for_mean]

if __name__ == "__main__":
    # spaces
    space_func = {'a': 10, 'b': -7.5}

    params = paragrid(model=test_func, space=space_func,
                      target='min', niter=30,
                      func_type = 'func')
    parameter, results = params.gradient_decent(lr = 1)
    param = params.score()
```
Here is a visual representation of the gradient descent
![Output sample](https://github.com/malteal/paragrid/blob/master/examples/figures/gradientdescent.gif)
## Authors

* **Malte Algren**
## Future improvements
- Add the missing description to the functions
- Add error checking for functions
- Fix gradient descent
- Add numba support

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

