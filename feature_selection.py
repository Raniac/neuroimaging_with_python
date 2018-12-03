def RFE_CV(optimal_model, X, y, list_features):
    print('Running RFE...')
    from sklearn.feature_selection import RFECV
    rfecv = RFECV(estimator=optimal_model, step=1, cv=10, scoring='accuracy', verbose=False)
    rfecv.fit(X, y)
    
    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Maximum cross-validation score: %.2f' % max(rfecv.grid_scores_))
    
    # select features with highest ranking
    list_selected_features = []
    for i, feat in enumerate(rfecv.ranking_):
        if feat == 1:
            list_selected_features.append(list_features[i]) # 2+i???
    print(list_selected_features)

    # Plot number of features VS. cross-validation scores
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

def f_score_CV(optimal_model, X, y, list_features):
    print('Running f score feature selection...')
    from sklearn.feature_selection import SelectFdr, f_classif
    X_new = SelectFdr(score_func=f_classif, alpha=0.01).fit_transform(X, y)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(optimal_model, X_new, y, scoring='accuracy', cv=10)
    print('Mean cross-validation score: %.2f' % scores.mean())

def t_score_CV(optimal_model, X, y, list_features):
    pass
    # import statsmodels.stats.weightstats as st
    # NC, SZ = X[:60], X[60:]