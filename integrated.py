import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def integrated_clf_model(model, data, k):
    
    from model_optimization import clf_model_optimization
    print('Running grid search...')
    gs_results = clf_model_optimization(model.model, data.X, data.y, k, model.param_grid)
    optimal_model = gs_results[0]
    print('The best parameter setting is: ' + str(gs_results[1]))
    print('The corresponding accuracy is: %.2f' % gs_results[2])

    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    list_weight_vectors = []
    accuracy = []
    sensitivity = []
    specificity = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    dict_split_data = data.data_k_split(k)
    optimal_model.probability = True
    c = 0
    for i in range(1, k+1):
        print('Iteration ' + str(i) + '...')
        train_X = dict_split_data['train_X_'+str(i)]
        train_y = dict_split_data['train_y_'+str(i)]
        test_X = dict_split_data['test_X_'+str(i)]
        test_y = dict_split_data['test_y_'+str(i)]

        print('Running without rfe...')
        optimal_model.fit(train_X, train_y)
        predictions = optimal_model.predict(test_X)

        if model.name in ['SVM', 'LR', 'LDA']:
            weight_vector = optimal_model.coef_
            # print('The weight vector is', weight_vector)
            list_weight_vectors.append(weight_vector)

        probas_ = optimal_model.predict_proba(test_X)

        print('Computing confusion matrix...')
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
        print('The confusion matrix is:', (tn, fp, fn, tp))
        cnf_accuracy = (tn + tp) / (tn + fp + fn + tp)
        print('The accuracy is: %.2f' % cnf_accuracy)
        accuracy.append(cnf_accuracy)
        cnf_sensitivity = tp / (tp + fn)
        print('The sensitivity is: %.2f' % cnf_sensitivity)
        sensitivity.append(cnf_sensitivity)
        cnf_specificity = tn / (tn + fp)
        print('The specificity is: %.2f' % cnf_specificity)
        specificity.append(cnf_specificity)

        fpr, tpr, _ = roc_curve(test_y, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (c, roc_auc))

        c += 1
        print('\n')

    print('Plotting mean ROC curve...')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    print('The mean fpr is:')
    print(mean_fpr)
    print('The mean tpr is:')
    print(mean_tpr)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print('The mean auc is:', mean_auc)
    print('The std auc is:', std_auc)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('results/' + 'ROC_curve_FE_CH_' + model.name + '_' + data.name + '.png', dpi=300)
    # plt.show()

    mean_accuracy = sum(accuracy) / len(accuracy)
    print('The mean accuracy: %.2f' % mean_accuracy)
    mean_sensitivity = sum(sensitivity) / len(sensitivity)
    print('The mean sensitivity: %.2f' % mean_sensitivity)
    mean_specificity = sum(specificity) / len(specificity)
    print('The mean specificity: %.2f' % mean_specificity)
    
    if model.name in ['SVM', 'LR', 'LDA']:
        sum_w_vecs = sum(list_weight_vectors)
        mean_w_vec = sum_w_vecs / k
        abs_w_vec = np.abs(mean_w_vec)
        feature_weight_dataframe = pd.DataFrame({'Feature': data.list_features, 'Weight': abs_w_vec[0]})
        feature_ranking = feature_weight_dataframe.sort_values('Weight', axis=0, ascending=False)
        print('The feature ranking sorted by weight is:')
        print(feature_ranking)
 
def integrated_rgs_model(model, data, k):
    print('Running gird search...')
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model.model, param_grid=model.param_grid, cv=k, scoring='neg_mean_absolute_error')
    gs.fit(data.X, data.y)
    optimal_model = gs.best_estimator_
    print('The best parameter setting is: ' + str(gs.best_params_))
    print('The corresponding MAE is: %.2f' % gs.best_score_)
    
    from sklearn.model_selection import cross_val_predict
    predictions = cross_val_predict(optimal_model, data.X, data.y, cv=10)
    original_predicted = pd.DataFrame({'Original': data.y, 'Predicted': predictions})
#     print(original_predicted)

    weight_vector = optimal_model.coef_
    abs_weight_vector = np.abs(weight_vector)
    feature_weight_dataframe = pd.DataFrame({'Feature': data.list_features, 'Weight': abs_weight_vector[0]})
    feature_ranking = feature_weight_dataframe.sort_values('Weight', axis=0, ascending=False)
    print('The feature ranking sorted by weight is:')
    print(feature_ranking)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy
    pearsonr, p = scipy.stats.pearsonr(data.y, predictions)
    print('The pearsonr and p are:', pearsonr, 'and', p)
    g = sns.jointplot(x='Original', y='Predicted', data=original_predicted, kind='reg', label='pearsonr = %.2f, p = %.4f' % (pearsonr, p))
    plt.legend(loc='upper right')
    g.savefig('results/' + 'Corr_FE_CH_' + model.name + '_' + data.name + '.png', dpi=300)
    plt.show()