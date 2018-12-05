import pandas as pd
import numpy as np

def data_acquisition():
    '''
    :type path: string of file path
    :type filename: string of filename
    :rtype X: dataframe X as feature matrix
    :rtype y: series y as label vector
    :rtype list_features: list of features
    '''
    info = pd.read_csv('~/Projects/neuroimaging_with_python/datasets/info.csv', encoding='gbk')
    y_all = info.GROUP[0:345].copy() # NC vs SZ
    y_all[266:345] = 1
    # y_sz = info.GROUP[205:345].copy() # FE vs CH
    # y_p = info.P_ALL[205:345].copy() # P_ALL
    # y_n = info.N_ALL[205:345].copy() # N_ALL
    # y_g = info.G_ALL[205:345].copy() # G_ALL
    # y_s = info.SCOREP1[205:345].copy() # SCOREP1

    path = '~/Projects/neuroimaging_with_python/datasets/classification/two_class/NC_SZ/'
    filename = 'ALFF_90.csv'
    df = pd.read_csv(path+filename, encoding='gbk')
    X = df.drop(['ID', 'GROUP'], axis=1)
    list_features = list(X.columns) # get the feature list
    
    # X = np.concatenate((X1, X2, X3, X4, X5), axis=1)
    # list_features = list_features1 + list_features2 + list_features3 + list_features4 + list_features5
    # print(X)

    # from sklearn import preprocessing
    # scaler = preprocessing.MinMaxScaler()
    # X = scaler.fit_transform(X)
    # X = preprocessing.scale(X) # standardization with respect to mean
    
    return X, y_all, list_features

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