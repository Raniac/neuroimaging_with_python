# neuroimaging_with_python

This is a repository consisting of the scripts working as analysis tools based on Machine Learning for neuroimaging data. It's for academic purpose only.

## How to use it?

- Change the path in the instance of **Logger** class to output results;
- Change the **FILENAME** and the **self.X** attribute in the instance of **Data** class in **main.py** to use different datasets, or even concatenate ones;
- Change the model class to instantiate to use different model.

## Updates on 2018.12.13

### What's new

- Added regression models;
- Plotted correlations between regression predictions and original y;
- Classification can now output feature weight vectors.

## Updates on 2018.12.10

### What's new

- Defined a Data class, its attributes include data name, feature matrix, label vector and list of features, its methods consists of preprocessing and split;
- Used ROC curve as one of the performance metrics;
- Defined model classes, their attributes include model instance, model name and grid search parameter settings.

### Notice

- Feature selection methods, e.g. f-score, are not completely done;
- ROC curves of different models are not plotted on a single figure.

## Updates on 2018.12.05

### What's new

- Separated **data_acquisition** and **model_optimization** from the original file;
- Used the label vectors from **info.csv**, avoiding creating too many datasets;
- Used **plt.savefig** to save figures generated.

### Notice

- Still no new preprocessing method used;
- A file-to-file IO model remains to be figured out.


## Updates on 2018.12.04

### What's new

- Completed the integrated_rgs_model, with neg_mean_absolute_error as performance metric. Relevance vector regression (RVR) remains to be configured;
- Run on different datasets with 345 subjects;
- Run on concatenated features to obtain higher accuracy.

### Notice

- Choose the right column to predict when using the regression models;
- It's not yet fully automated, thus it'd be better to modify the datasets manually;
- The accuracy is far from satisfactory, consider using different preprocessing methods.

## Updates on 2018.12.03

Refactoring - add **clf_models.py** and **rgs_models.py** to reduce the amount of code in **integrated.py**; each ***_models.py** contains the model initialization and parameter settings for grid search.

In the last version there was a mistake in line 205, causing index-out-of-range error,
```python
list_selected_features.append(list_features[2+i])
```
the '2+i' was non-sense and is now 'i'.

Updates also include adding LogisticRegression, LinearDiscriminantAnalysis, and KNeighborsClassifier models along with their grid search parameter settings. Moreover, f-score feature selection is also included.

## Updates on 2018.11.29

Committing the first version of scripts, including a integrated classification model and a integrated regression model (not completely done yet).

### Integrated classification model

Since an external validation loop will result in different parameter settings, doing grid search followed by feature selection will produce different model after each iteration. Therefore, it would be a wiser choice to do a grid search first outside the validation loop inside of which feature selection will be performed, hence can we select the optimal model and obtain the corresponding selected features with larger weights, along with the mean accuracy. Should mention that most of the models are not yet configured.

### Integrated regression model

It's not done so far while the main idea is to use **seaborn** to generate a bunch of figures showing correlations as well as the regression of PANSS scores.

### Errors encountered and solutions

Running the script **integrated.py** for the first time, an error like below occurred,
```bash
ModuleNotFoundError: No module named 'tkinter'
```
refer to [this site](https://www.jianshu.com/p/0baa9657377f) for solution, it seems that tkinter is not installed correctly.