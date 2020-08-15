# Paragrid

Paragrid is a simple parallized gridsearch, that are able to utilize all cpu core.
The main focus of this package is to reduce the lines of code, one has to write to find a good estimate for the parameters for a function.
This package works for most machine learning method as well as functions (see function section).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

Recent release:
```
pip install paragrid
```

To install the git codebase to add modifications:
```
git clone https://github.com/malteal/paragrid.git
```
### Usage

#### Function
Example for using paragrid to find the optimal parameters of a function
```python
from sklearn.datasets import load_boston
import numpy as np

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Parallel gridsearch
import paragrid

def test_func(X, y, std, learning_rate, n_estimators):
    mask = std<np.std(X, axis = 0)
    X = X[:,mask] ## setting restriction on std of columns
    reg_gpdt = GradientBoostingRegressor(loss = 'lad',
                                         learning_rate = learning_rate,
                                         n_estimators = n_estimators)
    return np.mean(cross_val_score(reg_gpdt, X, y, cv = 5))

# spaces
space_func = [{'std': [1, 20], 'learning_rate': [0.01, 0.1],
               'n_estimators': [2, 50]}]
# Regression
boston = load_boston()
X, y = boston.data, boston.target

reg_class = test_func
params = paragrid(model=reg_class, space=space_func,
                  X=X, y=y, target='min',
                  niter=0, func_type = 'func')
params.gridsearch()
param = params.score()
```
#### Classification
Example for using paragrid for classification in ML using scikit-optimize
```python
from sklearn.datasets import load_breast_cancer
from skopt.space import Real, Integer

# Classifiers
from lightgbm import LGBMClassifier

# Parallel gridsearch
import paragrid

# spaces
space_gpdt = [Integer(2, 50, name='max_depth'),
              Real(0.01, 0.1, "log-uniform", name='learning_rate'),
              Integer(2, 50, name='n_estimators')]
ncalls = 20

# Classification
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target    
lgbm_cls = LGBMClassifier()

params = paragrid(model=lgbm_cls, space=space_gpdt,
                                X=X, y=y, ncalls = ncalls, mtype = 'cls',
                                niter = 0)
params, results = params.gridsearch()
best_params = params.score()
```
#### Regression
Example for using paragrid for regression in ML with normal gridsearch.
Ex: learning_rate: from 0.01 to 0.1 and 10 points in between.
``` python
from sklearn.datasets import load_boston

# Classifiers
from sklearn.ensemble import GradientBoostingRegressor

# Parallel gridsearch
import paragrid

# spaces
space_gpdt = {'learning_rate': [0.01, 0.1, 10],
           'n_estimators': [2, 50, 10], 'loss' : ['ls', 'lad']}

# Regression
boston = load_boston()
X, y = boston.data, boston.target
reg_gpdt = GradientBoostingRegressor()

params = paragrid(model=reg_gpdt, space=space_gpdt,
                            X=X, y=y, target = 'min',
                            niter = 0)

params, results = params.gridsearch()

param_best = params.score()

```
## Authors

* **Malte Algren**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

