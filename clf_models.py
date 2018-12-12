class SVM_CLF():
    def __init__(self):
        from sklearn.svm import SVC
        self.model = SVC(
            C=1.0,
            kernel='linear'
        )
        self.name = 'SVM'
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    
    def rfe_selection(self):
        pass

class RF_CLF():
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            bootstrap=True,
            criterion='entropy',
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=5,
            n_estimators=10,
            random_state=0
        )
        self.name = 'RF'
        self.param_grid = {
            "max_depth": [3, 15],
            "min_samples_split": [3, 5, 10],
            "min_samples_leaf": [3, 5, 10],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
            "n_estimators": list(range(10, 50, 10))
        }

class LR_CLF():
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            random_state=0,
            solver='lbfgs',
            multi_class='multinomial'
        )
        self.name = 'LR'
        self.param_grid = {
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial', 'auto']
        }

class LDA_CLF():
    def __init__(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.model = LinearDiscriminantAnalysis(
            solver='svd'
        )
        self.name = 'LDA'
        self.param_grid = {
            'solver':['svd', 'lsqr']
        }

class KNN_CLF():
    def __init__(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(
            n_neighbors=10,
            algorithm='auto',
            p=1
        )
        self.name = 'KNN'
        self.param_grid = {
            'n_neighbors': [5, 6, 7, 8, 9, 10],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [0, 1, 2]
        }