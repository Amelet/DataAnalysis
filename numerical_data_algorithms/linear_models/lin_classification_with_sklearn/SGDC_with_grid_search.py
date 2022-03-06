import warnings
warnings.filterwarnings('ignore')
from sklearn import model_selection, datasets, linear_model, metrics
import numpy as np
import pandas as pd


# DATA SET
iris = datasets.load_iris()
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(iris.data, iris.target,
                                                                                     test_size=0.3, random_state=0)


# initialize CLASSIFIER, find its PARAMETERS, choose a few to search for their optimal values
classifier = linear_model.SGDClassifier(random_state=0, tol=1e-3)
print("classifiers keys: ", classifier.get_params().keys())
print("\tFor grid search we choose: loss, penalty, max_iter, alpha")
parameters_grid = {
    'loss': ['hinge', 'log', 'squared_hinge', 'squared_loss'],
    'penalty': ['l1', 'l2'],
    'max_iter': np.arange(5,10),
    'alpha': np.linspace(0.0001, 0.001, num=5)}


# This cross-validation object is a merge of StratifiedKFold and ShuffleSplit,
# which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.
cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


# GRID SEARCH
do_random = True
print("Do random gridsearch? ", do_random)
if do_random:
    randomized_grid_cv = model_selection.RandomizedSearchCV(classifier, parameters_grid, scoring='accuracy', cv=cv,
                                                            n_iter=20,
                                                            random_state=0)
    randomized_grid_cv.fit(train_data, train_labels)
    print("\tbest estimator: ", randomized_grid_cv.best_estimator_)
    print("\tbest score: ", randomized_grid_cv.best_score_)
    print("\tbest_params: ", randomized_grid_cv.best_params_)
else:
    grid_cv = model_selection.GridSearchCV(classifier, parameters_grid, scoring='accuracy', cv=cv)
    grid_cv.fit(train_data, train_labels)
    print("\tbest estimator: ", grid_cv.best_estimator_)
    print("\tbest score: ", grid_cv.best_score_)
    print("\tbest_params: ", grid_cv.best_params_)