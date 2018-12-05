import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from data_acquisition import data_acquisition
from clf_models import *
from rgs_models import *
from feature_selection import *
from model_optimization import *

import sys
class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def integrated_clf_model(model, X, y, k, param_grid, list_features):
    print('Running grid search...')
    gs_results = clf_model_optimization(model, X, y, k, param_grid)
    optimal_model = gs_results[0]
    print('The best parameter setting is: ' + str(gs_results[1]))
    print('The corresponding accuracy is: %.2f' % gs_results[2])

    RFE_CV(optimal_model, X, y, list_features)
    # f_score_CV(optimal_model, X, y, list_features)
 
def integrated_rgs_model(model, X, y, k, param_grid, list_features):
    print('Running gird search...')
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_absolute_error')
    gs.fit(X, y)
    # optimal_model = gs.best_estimator_
    print('The best parameter setting is: ' + str(gs.best_params_))
    print('The corresponding mae is: %.2f' % gs.best_score_)

if __name__ == "__main__":
    sys.stdout = Logger('results/results_test_181129.txt') # console to file
    X, y, list_features = data_acquisition()
    integrated_clf_model(svm_clf(), X, y, 10, param_grid_svm, list_features)
    # integrated_rgs_model(l2_rgs(), X, y, 10, param_grid_l2, list_features)