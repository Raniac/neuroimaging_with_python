def svm_clf():
    from sklearn.svm import SVC
    model = SVC(
        C=1.0,
        kernel='linear'
    )
    return model

param_grid_svm = {
    'C': [0.01, 1, 10, 100],
    'kernel': ['linear']
}

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

param_grid_rf = {
    "max_depth": [3, 15],
    "min_samples_split": [3, 5, 10],
    "min_samples_leaf": [3, 5, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "n_estimators": list(range(10, 50, 10))
}

def lr_clf():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        multi_class='multinomial'
    )
    return model

param_grid_lr = {
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'multi_class': ['ovr', 'multinomial', 'auto']
}

def lda_clf():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis(
        solver='svd'
    )
    return model

param_grid_lda = {
    'solver':['svd', 'lsqr']
}

def knn_clf():
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(
        n_neighbors=10,
        algorithm='auto'
    )
    return model

param_grid_knn = {
    'n_neighbors': [5, 10, 50],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}