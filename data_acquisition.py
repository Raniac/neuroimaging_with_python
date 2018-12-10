import pandas as pd
import numpy as np

WORKING_DIR = '~/Projects/neuroimaging_with_python/'
DATASETS_DIR = WORKING_DIR + 'datasets/'

info = pd.read_csv(DATASETS_DIR + 'info.csv', encoding='gbk')
y_all = info.GROUP[0:345].copy() # NC vs SZ
y_all[266:345] = 1
y_sz = info.GROUP[205:345].copy() # FE vs CH
y_p = info.P_ALL[205:345].copy() # P_ALL
y_n = info.N_ALL[205:345].copy() # N_ALL
y_g = info.G_ALL[205:345].copy() # G_ALL
y_s = info.SCOREP1[205:345].copy() # SCOREP1

# Choose the dataset to use
PATH = DATASETS_DIR + 'NC_SZ/'
FILENAME = 'MD_90' # without file extension

# Choose the corresponding y
y = y_all

class Data():
    def __init__(self):
        self.name = FILENAME
        self.X = pd.read_csv(PATH + FILENAME + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1)
        self.y = y
        self.list_features = list(self.X.columns)

    def data_preprocessing(self):
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = preprocessing.scale(self.X)

    def data_split(self):
        from sklearn.model_selection import train_test_split
        train_X, test_X, train_y, test_y = train_test_split(self.X, self.y, random_state=0)
        return train_X, test_X, train_y, test_y

    def data_k_split(self, k):
        '''
        :type k: number of fold to split
        :rtype dict_split: dictionary containing k np.array of 
                            train_X, train_y, test_X, test_y
        '''
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=k)
        dict_split = {} # used to store the train/test pair
        idx = 1
        for train_idx, test_idx in skf.split(self.X, self.y):
            train_X_tmp = [] # used to store train X
            train_y_tmp = [] # used to store train y
            test_X_tmp = [] # used to store test X
            test_y_tmp = [] # used to store test y

            # generate training set
            list_train_idx = list(train_idx) # from np.array to list
            for t1 in range(0, len(list_train_idx)):
                num_tr = list_train_idx[t1]
                train_X_tmp.append(list(self.X.iloc[num_tr, :]))
                train_y_tmp.append(list(self.y)[num_tr])
            dict_split['train_X_'+str(idx)] = np.array(train_X_tmp)
            dict_split['train_y_'+str(idx)] = np.array(train_y_tmp)

            # generate testing set
            list_test_idx = list(test_idx) # from np.array to list
            for t2 in range(0, len(list_test_idx)):
                num_ts = list_test_idx[t2]
                test_X_tmp.append(list(self.X.iloc[num_ts]))
                test_y_tmp.append(list(self.y)[num_ts])
            dict_split['test_X_'+str(idx)] = np.array(test_X_tmp)
            dict_split['test_y_'+str(idx)] = np.array(test_y_tmp)
            
            idx += 1
        return dict_split