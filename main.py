from datasets import *
from data_acquisition import Data
from clf_models import *
from rgs_models import *
from integrated import *

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
    sys.stdout = Logger('results/' + time.strftime('%y%m%d') + '_' + 'P3_SVM_fMRI_246.txt') # set result filename
    my_data = Data(fMRI, X_fMRI, y_panss) # instantiate my data object
    my_data.data_preprocessing()
    my_model = SVM_CLF() # instantiate my classification model object
    integrated_clf_model(my_model, my_data, 10)
    # my_model = SVR_RGS() # instantiate my regression model object
    # integrated_rgs_model(my_model, my_data, 10)