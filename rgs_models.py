class OLS_RGS():
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.name = 'OLS'
        self.param_grid = {}

class L1_RGS():
    def __init__(self):
        from sklearn.linear_model import Lasso
        self.model = Lasso()
        self.name = 'LASSO'
        self.param_grid = {
            'alpha': list(range(4, 21, 2))
        }

class L2_RGS():
    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge()
        self.name = 'RIDGE'
        self.param_grid = {
            'alpha': list(range(0, 21, 2))
        }

class SVR_RGS():
    def __init__(self):
        from sklearn.svm import SVR
        self.model = SVR()
        self.name = 'SVR'
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }

class EN_RGS():
    def __init__(self):
        from sklearn.linear_model import ElasticNet
        self.model = ElasticNet()
        self.name = 'ElasticNet'
        self.param_grid = {
            'alpha': list(range(4, 21, 2)),
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }