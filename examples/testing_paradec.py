#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:43:47 2020

@author: malte
"""

# Parallel gridsearch
# from paragrid import paragrid
import numpy as np
# Help packages
from sklearn.datasets import load_breast_cancer
import concurrent.futures
# Classifiers
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
import sys
sys.path.insert(1,'../paragrid')
#%%

from paradec import parallel
@parallel
def ml_model(X,y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return np.sum(y_pred==y_test)/len(y_test)
# spaces
space_gpdt = {'learning_rate': [0.001, 0.1, ],
              'n_estimators': [2, 70, 5],
              'max_depth': [2, 50, 4]}


# Classification
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target    

args  = [[X,y,LGBMClassifier(n_estimators=i)] for i in [5,10,15,25]]
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(ml_model, args)
for i in results:
    print(i)