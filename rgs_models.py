def ols_rgs():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    return model

param_grid_ols = {}

def l1_rgs():
    from sklearn.linear_model import Lasso
    model = Lasso(
        alpha=1.0
    )
    return model

param_grid_l1 = {
    'alpha': [0.01, 0.1, 1.0, 10]
}

def l2_rgs():
    from sklearn.linear_model import Ridge
    model = Ridge(
        alpha=1.0
    )
    return model

param_grid_l2 = {
    'alpha': [0.01, 0.1, 1.0, 10]
}

def svr_rgs():
    from sklearn.svm import SVR
    model = SVR(
        C=1.0,
    )
    return model

param_grid_svr = {
    'C': [0.01, 0.1, 1, 10, 100]
}

def rvr_rgs():
        pass

param_grid_rvr = {}