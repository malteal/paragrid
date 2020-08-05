# Paragrid

Paragrid is a simple parallized gridsearch, that are able to utilize all cpu core.
The main focus of this package is to reduce the lines of code, one has to write to find a good estimate for the hyperparameters.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Classification
```python
from sklearn.datasets import load_breast_cancer ## data
from skopt.space import Real, Integer ## space
from lightgbm import LGBMClassifier ## classifier

# Parallel gridsearch
from functions import paragrid

# spaces
space_gpdt = [Integer(2, 50, name='max_depth'),
              Real(0.01, 0.1, "log-uniform", name='learning_rate'),
              Integer(2, 50, name='n_estimators')]
              
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target    
lgbm_cls = LGBMClassifier()

params = paragrid(model=lgbm_cls, space=space_gpdt,
                         X=X, y=y, ncalls = 20, mtype = 'cls',
                         niter = 0)
params, results = params.gridsearch()

```
### Installing

```
pip install paragrid
```

## Authors

* **Malte Algren**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

