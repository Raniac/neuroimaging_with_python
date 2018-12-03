def ols_rgs():
    pass

param_grid_ols = {}

def l1_rgs():
    pass

param_grid_l1 = {}

def l2_rgs():
    pass

param_grid_l2 = {}

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