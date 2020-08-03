# Paragrid

Paragrid is a simple parallized gridsearch, that are able to utilize all cpu core.
The main focus of this package is to reduce the lines of code, one has to write to find a good estimate for the hyperparameters.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

params = hyper_parameter_tuning(model=lgbm_cls, space=space_gpdt,
                         X=X, y=y, ncalls = 20, mtype = 'cls',
                         niter = 0)
params, results = params.gridsearch()

```

### Prerequisites

What things you need to install the software and how to install them

```
pip install skopt
pip install paragrid
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Malte Algren**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

