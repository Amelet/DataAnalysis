from matplotlib.colors import ListedColormap
from sklearn import model_selection, datasets, linear_model, metrics


# SNIPPETS FOR LINEAR CLASSIFICATION
# model
def get_classifier_trained(linclass_type, train_data, test_data):
    """Creates an object of selected classifier.
    Then trains classifier"""
    if linclass_type == "ridge":
        classifier_m = linear_model.RidgeClassifier(random_state=1)     # 1. create object
    elif linclass_type == "logistic":
        classifier_m = linear_model.LogisticRegression(random_state=1)
    classifier_m.fit(train_data, train_labels)                               # 2. train classifier's object
    coefficients = classifier_m.coef_
    intercept = classifier_m.intercept_
    return classifier_m, coefficients, intercept


# model's metrics
def get_metrics(classifier_m, test_data, test_labels, data):
    """Predicts output from the test data set.
    Measures accuracy of the model with cross-validation (two CV strategies)"""
    predictions = classifier_m.predict(test_data)                          # 3. predict
    accuracy_sc = metrics.accuracy_score(test_labels, predictions)
    scoring_m = model_selection.cross_val_score(classifier_m, data[0], data[1],
                                                scoring='accuracy', cv=10)
    scoring_with_strategy = model_selection.cross_val_score(classifier_m, data[0], data[1],
                                                            scoring='accuracy', cv=cv_strategy)
    return predictions, accuracy_sc, scoring_m, scoring_with_strategy


# USER'S CHOICE: lin_classifier type
def choose_linclass_type():
    """Takes input from user"""
    while True:
        linclass_type = input("""
Which linear classifier model to use?
    Options:
    - logistic
    - ridge.
Please, type here one of two:    
    """)
        if linclass_type in linclass_types:
            break
        else:
            print(f"Please type one of the offered options. You typed '{linclass_type}'")
    return linclass_type


# CHOOSE MODEL (users choice)
linclass_types = ["logistic", "ridge"]
linclass_type = choose_linclass_type()


# generate data
blobs = datasets.make_blobs(centers=2, cluster_std=5.5, random_state=1)
colors = ListedColormap(['red', 'blue'])
# fig = plt.figure(figsize(8, 8))
# plt.scatter([x[0] for x in blobs[0]], [x[1] for x in blobs[0]], c=blobs[1], cmap=colors)


# train test
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(blobs[0], blobs[1],
                                                                                    test_size=0.3,
                                                                                    random_state=1)

# select scorer and cross-validation strategy
scorer = metrics.make_scorer(metrics.accuracy_score)
cv_strategy = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=2)
cv_strategy.get_n_splits(blobs[1])


# Linear model
classifier_m, coefficients, intercept = get_classifier_trained(linclass_type, train_data, test_data)
predictions, accuracy_sc, scoring_m, scoring_with_strategy = get_metrics(classifier_m, test_data, test_labels, blobs)


# prints results
for sc, cv_type in zip([scoring_m, scoring_with_strategy], ["CV=10", "CV with strategy"]):
    print(f'''{linclass_type} classifier, cross-validation strategy {cv_type}, metrics = accuracy:
        mean: {sc.mean()},
        max: {sc.max()},
        min: {sc.min()},
        std: {sc.std()}''')
