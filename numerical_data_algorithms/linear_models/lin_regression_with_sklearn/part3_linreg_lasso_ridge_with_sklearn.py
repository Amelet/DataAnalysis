import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn import model_selection, datasets, linear_model, metrics
np.set_printoptions(precision=3, suppress=True)


print("""
# The algorithm calculates linear regression coefficients for three types of model:
- linear regression (does not handle well collinearity in matrix, over-fits)
- lasso regression (Lasso is better for penalizing coefficients for uninformative features)
- ridge regression (penalizes as well)
# Ridge and Lasso regression are used to reduce model's over-fitting.

# Ridge Regression : the cost function is penalized by the coefficients' square of the magnitude
# multiplied by the regularization coefficient λ.
# For low λ the model is close to linear regression model.

# Lasso regression:  the cost function is penalized by the coefficients' magnitude
# multiplied by λ""")


# LINEAR REGRESSION function:
def get_models_coefs(train_data, train_labels, linreg_type, regularization_coef):
    """Accepts a choice of linreg model (normal linreg, lasso, ridge)
    and chosen regularization coefficient (for lasso, ridge).
    Workflow:
    1) Creates linreg object.
    2) Fits model.
    3) Returns model's coefficients"""
    if linreg_type == "linreg":
        regressor_m = LinearRegression()                       # 1. Linreg object is created
    elif linreg_type == "lasso":
        regressor_m = Lasso(random_state=3, alpha=regularization_coef)
    elif linreg_type == "ridge":
        regressor_m = Ridge(random_state=3, alpha=regularization_coef)
    regressor_m.fit(train_data, train_labels)              # 2. Model training
    coefficients = regressor_m.coef_                       # 3. Regression coefficients obtained
    return coefficients, regressor_m


def make_predictions(model, test_data, test_labels):
    predictions = model.predict(test_data)
    error = metrics.mean_absolute_error(test_labels, predictions)
    return predictions, error


def scoring_cross_validation(model, data, target):
    linear_scoring = model_selection.cross_val_score(model, data, target, scoring='neg_mean_absolute_error',
                                                     cv=10)
    # scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=True)
    # linear_scoring = model_selection.cross_val_score(linear_regressor, data, target, scoring=scorer, cv=10)
    return linear_scoring.mean(), linear_scoring.std()


# USERS CHOICE functions:
# choose linreg type (normal, lasso, ridge)
# and regularization coefficient (for lasso, ridge)


# linreg type
def choose_linreg_type():
    while True:
        linreg_type = input("""
Which linreg model to use?
    Options:
    - linreg
    - lasso
    - ridge.
Please, type here one of three:    
    """)
        if linreg_type in linreg_types:
            break
        else:
            print(f"Please type one of the offered options. You typed '{linreg_type}'")
    return linreg_type


# regularization coefficient choice
def choose_regularization_coef():
    while True:
        if linreg_type == "linreg":
            regularization_coef = None
            break
        elif linreg_type in ["lasso", "ridge"]:
            try:
                regularization_coef = input(f"""
You have chosen {linreg_type} that requires regularization coefficient.
    For example:  1
Type coefficient here:   """)
                regularization_coef = float(regularization_coef)
                break
            except ValueError:
                print(f"Type a single digit (int or float), you typed '{regularization_coef}'")
    return regularization_coef


# CHOOSE MODEL (users choice)
linreg_types = ["linreg", "lasso", "ridge"]
linreg_type = choose_linreg_type()
regularization_coef = choose_regularization_coef()


# PREPARE DATA (load, shuffle, scale it)
df = pd.read_csv("bikes_rent.csv")
df_shuffled = shuffle(df, random_state=123)
X = scale(df_shuffled[df_shuffled.columns[:-1]])
y = df_shuffled.iloc[:, -1]
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(X, y, test_size=0.3)


# DO LINREG fitting
print(f"Model is {linreg_type}, regularisation coefficient is {regularization_coef}"
      f"\nFitting model ...")
variables_list = df.columns.values[:-1]
output_vals = df.columns.values[-1]
coefficients, regressor_model = get_models_coefs(train_data, train_labels, linreg_type, regularization_coef)
print("Model is fitted\n")


# PREDICTIONS
predictions, error = make_predictions(regressor_model, test_data, test_labels)
print("model's error = ", error)


# LINEAR SCORING
linscor_mean, linscor_std = scoring_cross_validation(regressor_model, X, y)
print(f"linear scoring (mean) = {linscor_mean}, (std) = {linscor_std}")


# Print results:
try:
    if len(coefficients)>0:
        print("Model's coefficients:")
        for variable, coef in zip(df.columns, coefficients):
            print(f"\t{variable}, {coef}")
except:
    print("Coefficients were not calculated")