def clf_model_optimization(model, X, y, k, param_grid):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='accuracy')
    gs.fit(X, y)
    return (gs.best_estimator_, gs.best_params_, gs.best_score_)

def rgs_model_optimization(model, X, y, k, param_grid):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_absolute_error')
    gs.fit(X, y)
    return (gs.best_estimator_, gs.best_params_, gs.best_score_)