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

    list_feature_sets = []
    accuracy = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    dict_split_data = data.data_k_split(k)
    optimal_model.probability = True
    for i in range(1, k+1):
        print('iteration ' + str(i) + '...')
        train_X = dict_split_data['train_X_'+str(i)]
        train_y = dict_split_data['train_y_'+str(i)]
        test_X = dict_split_data['test_X_'+str(i)]
        test_y = dict_split_data['test_y_'+str(i)]

        print('running recursive feature elimination...')
        from sklearn.feature_selection import RFE
        RFE_selector = RFE(estimator=optimal_model, n_features_to_select=40, step=1)
        RFE_selector.fit(train_X, train_y)        
        predictions_i = RFE_selector.predict(test_X)
        accuracy_i = 1 - sum(abs(test_y - predictions_i))/len(predictions_i)
        print('accuracy after rfe: %.2f' % accuracy_i)
        accuracy.append(accuracy_i)

        print('generating selected feature list...')
        list_selected_features_i = []
        for i, feat in enumerate(RFE_selector.ranking_):
            if feat == 1:
                list_selected_features_i.append(data.list_features[i])
        print('selected features: ' + str(list_selected_features_i))
        list_feature_sets.append(set(list_selected_features_i))

        print('\n')

        probas_ = RFE_selector.predict_proba(test_X)
        fpr, tpr, _ = roc_curve(test_y, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
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
    plt.savefig('results/' + 'ROC_curve_' + model.name + '_' + data.name + '.png', dpi=300)
    plt.show()

    mean_accuracy = sum(accuracy)/len(accuracy)
    print('Mean accuracy: %.2f' % mean_accuracy)
    
    common_selected_features = []
    tmp = list_feature_sets[0]
    for set_i in list_feature_sets:
        common_selected_features = set_i.intersection(tmp)
        tmp = set_i
    print('common selected features are: ' + str(common_selected_features))

    # RFE_CV(optimal_model, data.X, data.y, data.list_features)
    # f_score_CV(optimal_model, X, y, list_features)
 
# def integrated_rgs_model(model, data, k, param_grid):
#     print('Running gird search...')
#     from sklearn.model_selection import GridSearchCV
#     gs = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_absolute_error')
#     gs.fit(data.X, data.y)
#     # optimal_model = gs.best_estimator_
#     print('The best parameter setting is: ' + str(gs.best_params_))
#     print('The corresponding mae is: %.2f' % gs.best_score_)