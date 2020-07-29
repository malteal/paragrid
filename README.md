# OP Gridsearch




´´´ python
from sklearn.datasets import load_boston
from skopt.space import Real, Integer
# Classifiers
from sklearn.ensemble import GradientBoostingRegressor

from functions import hyper_parameter_tuning
# Data
boston = load_boston()
X, y = boston.data, boston.target

# Classifiers
reg_gpdt = GradientBoostingRegressor(loss = 'lad')

# spaces
space_gpdt = [Integer(1, 20, name='max_depth'),
           Real(10**-3, 1, "log-uniform", name='learning_rate'),
          # Real(0.1, 1, name='subsample'),
          # Integer(2, X.shape[1], name='max_features'),
           Integer(3, 50, name='n_estimators')
          ]

ncalls = 100
params = hyper_parameter_tuning(model=reg_cls_gpdt, space=space_gpdt,
                                X=X, y=y, ncalls = ncalls, mtype = 'res')
´´´
