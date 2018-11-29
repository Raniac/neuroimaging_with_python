import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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
    df = pd.read_csv(path+filename, encoding='gbk') # type of df is dataframe
    # print(df.describe()) # print the summary of the dataframe
    # print(df.columns) # print the indices of the columns
    # df = df.dropna(axis=1) # drop columns without value
    # df = df.dropna(axis=0) # drop rows without value
    y = df.group # use the dot notation to select the column to predict
    X = df.drop(['ID', 'group'], axis=1)
    # selected_features = [] # select features to use
    # X = X[selected_features]
    list_features = list(X.columns) # get the feature list
    
    from sklearn import preprocessing
    X = preprocessing.scale(X) # standardization with respect to mean
    
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

# model initialization
def svm_clf():
    from sklearn.svm import SVC
    model = SVC(
        C=1.0,
        kernel='linear'
    )
    return model
def rf_clf():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        bootstrap=True,
        criterion='entropy',
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=5,
        n_estimators=10,
        random_state=0
    )
    return model
def lr_clf():
    pass
def lda_clf():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis(
        solver='svd'
    )
    return model
def knn_clf():
    pass
def ols_rgs():
    pass
def l1_rgs():
    pass
def l2_rgs():
    pass
def svr_rgs():
    from sklearn.svm import SVR
    model = SVR(
        C=1.0,
    )
    return model
def rvr_rgs():
        pass
#/ model initialization

# grid search parameter setting
param_grid_svm = {
    'C': [0.01, 1, 10, 100],
    'kernel': ['linear']
}
param_grid_rf = {
    "max_depth": [3, 15],
    "min_samples_split": [3, 5, 10],
    "min_samples_leaf": [3, 5, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "n_estimators": list(range(10, 50, 10))
}
param_grid_lda = {
    'solver':['svd']
}
param_grid_knn = {}
param_grid_ols = {}
param_grid_l1 = {}
param_grid_l2 = {}
param_grid_svr = {
    'C': [0.01, 0.1, 1, 10, 100]
}
param_grid_rvr = {}
#/ grid search parameter setting

# classification model optimization
def clf_model_optimization(model, X, y, k, param_grid):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='accuracy')
    gs.fit(X, y)
    return (gs.best_estimator_, gs.best_params_, gs.best_score_)
#/ classification model optimization

# regression model optimization
def rgs_model_optimization(model, X, y, k, param_grid):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_absolute_error')
    gs.fit(X, y)
    return (gs.best_estimator_, gs.best_params_, gs.best_score_)
#/ regression model optimization

# feature selection
def RFE_selection(model, train_X, train_y, test_X, test_y):
    pass
def f_score_selection():
    pass
def t_score_selection():
    pass
#/ feature selection

# integrated classification model
def integrated_clf_model(model, X, y, k, param_grid, list_features):
    
    print('running grid search...')
    gs_results = clf_model_optimization(model, X, y, k, param_grid)
    optimal_model = gs_results[0]
    print('The best parameter setting is: ' + str(gs_results[1]))
    print('The corresponding accuracy is: %.2f' % gs_results[2])

    from sklearn.feature_selection import RFECV
    rfecv = RFECV(estimator=optimal_model, step=1, cv=10, scoring='accuracy', verbose=False)
    # step:If greater than or equal to 1, then 'step' corresponds to the (integer)
    # number of features to remove at each iteration.
    # cv: determine the cross-validation splitting strategy
    rfecv.fit(X, y)
    
    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Maximum cross-validation score: %.2f' % max(rfecv.grid_scores_))
    
    # select features with highest ranking
    list_selected_features = []
    for i, feat in enumerate(rfecv.ranking_):
        if feat == 1:
            list_selected_features.append(list_features[2+i])
    print(list_selected_features)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    # list_feature_sets = []
    # accuracy = []
    # dict_split_data = split_k_fold(k, X, y)
    # for i in range(1, k+1):
    #     print('iteration ' + str(i) + '...')
    #     train_X = dict_split_data['train_X_'+str(i)]
    #     train_y = dict_split_data['train_y_'+str(i)]
    #     test_X = dict_split_data['test_X_'+str(i)]
    #     test_y = dict_split_data['test_y_'+str(i)]

    #     # feature selection -> RFE
    #     print('running recursive feature elimination...')
    #     from sklearn.feature_selection import RFE
    #     RFE_selector = RFE(estimator=optimal_model, n_features_to_select=100, step=1)
    #     RFE_selector.fit(train_X, train_y)        
    #     predictions_i = RFE_selector.predict(test_X)
    #     accuracy_i = 1 - sum(abs(test_y - predictions_i))/len(predictions_i)
    #     print('accuracy after rfe: %.2f' % accuracy_i)
    #     accuracy.append(accuracy_i)

    #     print('generating selected feature list...')
    #     list_selected_features_i = []
    #     for i, feat in enumerate(RFE_selector.ranking_):
    #         if feat == 1:
    #             list_selected_features_i.append(list_features[i])
    #     print('selected features: ' + str(list_selected_features_i))
    #     list_feature_sets.append(set(list_selected_features_i))

    #     print('\n')

    # mean_accuracy = sum(accuracy)/len(accuracy)
    # print('Mean accuracy: %.2f' % mean_accuracy)
    
    # common_selected_features = []
    # tmp = list_feature_sets[0]
    # for set_i in list_feature_sets:
    #     common_selected_features = set_i.intersection(tmp)
    #     tmp = set_i
    # print('common selected features are: ' + str(common_selected_features))
#/ integrated classification model

# integrated regression model
def integrated_rgs_model(model, X, y, k, param_grid, list_features):
    pass
#/ integrated regression model

# main function
if __name__ == "__main__":
    sys.stdout = Logger('results/results_test_181129.txt')
    path = '~/Projects/neuroimaging_with_python/datasets/'
    filename = 'mwc1_AAL90_morph_features.csv'
    X, y, list_features = data_acquisition(path, filename)
    integrated_clf_model(svm_clf(), X, y, 10, param_grid_svm, list_features)
    # integrated_rgs_model(svr_rgs(), X, y, 10, param_grid_svr, list_features)
#/ main function