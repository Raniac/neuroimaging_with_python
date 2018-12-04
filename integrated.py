import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from clf_models import *
from rgs_models import *
from feature_selection import *

# log configuration
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
#/ log configuration

# data acquisition and manipulation
def data_acquisition(path, filename):
    '''
    :type path: string of file path
    :type filename: string of filename
    :rtype X: dataframe X as feature matrix
    :rtype y: series y as label vector
    :rtype list_features: list of features
    '''
    df = pd.read_csv(path+filename, encoding='gbk')
    y = df.GROUP # use the dot notation to select the column to predict
    X = df.drop(['ID', 'GROUP'], axis=1)
    list_features = list(X.columns) # get the feature list
    
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    # X = preprocessing.scale(X) # standardization with respect to mean
    
    return X, y, list_features

def split_k_fold(k, X, y):
    '''
    :type k: number of fold to split
    :type X: input data matrix
    :type y: input label series
    :rtype dict_split: dictionary containing k np.array of 
                       train_X, train_y, test_X, test_y
    '''
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k)
    dict_split = {} # used to store the train/test pair
    idx = 1
    for train_idx, test_idx in skf.split(X, y):
        train_X_tmp = [] # used to store train X
        train_y_tmp = [] # used to store train y
        test_X_tmp = [] # used to store test X
        test_y_tmp = [] # used to store test y

        # generate training set
        list_train_idx = list(train_idx) # from np.array to list
        for t1 in range(0, len(list_train_idx)):
            num_tr = list_train_idx[t1]
            train_X_tmp.append(list(X[num_tr, :]))
            train_y_tmp.append(list(y)[num_tr])
        dict_split['train_X_'+str(idx)] = np.array(train_X_tmp)
        dict_split['train_y_'+str(idx)] = np.array(train_y_tmp)

        # generate testing set
        list_test_idx = list(test_idx) # from np.array to list
        for t2 in range(0, len(list_test_idx)):
            num_ts = list_test_idx[t2]
            test_X_tmp.append(list(X[num_ts, :]))
            test_y_tmp.append(list(y)[num_ts])
        dict_split['test_X_'+str(idx)] = np.array(test_X_tmp)
        dict_split['test_y_'+str(idx)] = np.array(test_y_tmp)
        
        idx += 1
    return dict_split
#/ data acquisition and manipulation

# classification model optimization
def clf_model_optimization(model, X, y, k, param_grid):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='accuracy')
    gs.fit(X, y)
    return (gs.best_estimator_, gs.best_params_, gs.best_score_)
#/ classification model optimization

# integrated classification model
def integrated_clf_model(model, X, y, k, param_grid, list_features):
    print('Running grid search...')
    gs_results = clf_model_optimization(model, X, y, k, param_grid)
    optimal_model = gs_results[0]
    print('The best parameter setting is: ' + str(gs_results[1]))
    print('The corresponding accuracy is: %.2f' % gs_results[2])

    RFE_CV(optimal_model, X, y, list_features)
    # f_score_CV(optimal_model, X, y, list_features)
#/ integrated classification model

# integrated regression model
def integrated_rgs_model(model, X, y, k, param_grid, list_features):
    print('Running gird search...')
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_absolute_error')
    gs.fit(X, y)
    # optimal_model = gs.best_estimator_
    print('The best parameter setting is: ' + str(gs.best_params_))
    print('The corresponding mae is: %.2f' % gs.best_score_)
#/ integrated regression model

# main function
if __name__ == "__main__":
    sys.stdout = Logger('results/results_test_181129.txt')
    path = '~/Projects/neuroimaging_with_python/datasets/classification/two_class/NC_SZ/'
    filename = 'ALFF_90.csv'
    # filename2 = 'DC_90.csv'
    # filename3 = 'GMV_90.csv'
    # filename4 = 'ReHo_90.csv'
    # filename5 = 'WMV_90.csv'
    X, y, list_features = data_acquisition(path, filename)
    # X2, y, list_features2 = data_acquisition(path, filename2)
    # X3, y, list_features3 = data_acquisition(path, filename3)
    # X4, y, list_features4 = data_acquisition(path, filename4)
    # X5, y, list_features5 = data_acquisition(path, filename5)
    # X = np.concatenate((X1, X2, X3, X4, X5), axis=1)
    # list_features = list_features1 + list_features2 + list_features3 + list_features4 + list_features5
    print(X.shape)
    print(X)
    integrated_clf_model(svm_clf(), X, y, 10, param_grid_svm, list_features)
    # integrated_rgs_model(ols_rgs(), X, y, 10, param_grid_ols, list_features)
#/ main function