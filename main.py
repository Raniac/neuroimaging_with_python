from data_acquisition import Data
from clf_models import *
from rgs_models import *
from integrated import *

import pandas as pd
import numpy as np

WORKING_DIR = '~/Projects/neuroimaging_with_python/'
DATASETS_DIR = WORKING_DIR + 'datasets/'

info = pd.read_csv(DATASETS_DIR + 'info.csv', encoding='gbk')
y_all = info.GROUP[0:345].copy() # NC vs SZ
y_all[266:345] = 1
y_sz = info.GROUP[205:345].copy() # FE vs CH
y_sz[y_sz == 1] = 0
y_sz[y_sz == 2] = 1
y_p = info.P_ALL[205:345].copy() # P_ALL
y_n = info.N_ALL[205:345].copy() # N_ALL
y_g = info.G_ALL[205:345].copy() # G_ALL
y_s = info.SCOREP1[205:345].copy() # SCOREP1

# Choose the dataset to use
FILENAME1 = 'T1_246'
# X1 = pd.read_csv(DATASETS_DIR + FILENAME1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
X1 = pd.read_csv(DATASETS_DIR + FILENAME1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
FILENAME2 = 'fMRI_246'
# X2 = pd.read_csv(DATASETS_DIR + FILENAME2 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
X2 = pd.read_csv(DATASETS_DIR + FILENAME2 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
FILENAME3 = 'DTI_246'
# X3 = pd.read_csv(DATASETS_DIR + FILENAME3 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
X3 = pd.read_csv(DATASETS_DIR + FILENAME3 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
FILENAME = 'COMB_246'
# X = pd.concat([X1, X2, X3], axis=1)
X = pd.concat([X1, X2, X3], axis=1) # FE_CH

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

if __name__ == "__main__":
    sys.stdout = Logger('results/FE_CH_EN_' + FILENAME + '_181213.txt')
    my_data = Data(FILENAME, X, y_s) # instantiate my data object
    my_data.data_preprocessing()
    my_model = EN_RGS()
    # integrated_clf_model(my_model, my_data, 10)
    integrated_rgs_model(my_model, my_data, 10)