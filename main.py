from data_acquisition import Data
from clf_models import SVM_CLF
from integrated import integrated_clf_model

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
    sys.stdout = Logger('results/results_test_181129.txt') # console to file
    my_data = Data() # instantiate my data object
    svm_clf = SVM_CLF()
    integrated_clf_model(svm_clf, my_data, 10)
    # integrated_rgs_model(l2_rgs(), my_data, 10, param_grid_l2)