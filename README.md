# neuroimaging_with_python

This is a repository consisting of the scripts working as analysis tools based on Machine Learning for neuroimaging data. It's for academic purpose only.

## Updates on 2018.11.29:

Committing the first version of scripts, including a integrated classification model and a integrated regression model (not completely done yet).

### Integrated classification model

Since an external validation loop will results in different parameter settings, doing grid search followed by feature selection will produce different model after each iteration. Therefore, it would be a wiser choice to do a grid search first outside the validation loop inside of which feature selection will be performed, hence can we select the optimal model and obtain the corresponding selected features with larger weights, along with the mean accuracy. Should mention that most of the models are not yet configured.

### Integrated regression model

It's not done so far while the main idea is to use **seaborn** to generate a bunch of figures showing correlations as well as the regression of PANSS scores.

### Errors encountered and solutions

Running the script **integrated.py** for the first time, an error like below occurred,
```bash
ModuleNotFoundError: No module named 'tkinter'
```
refer to [this site](https://www.jianshu.com/p/0baa9657377f) for solution, it seems that tkinter is not installed correctly.