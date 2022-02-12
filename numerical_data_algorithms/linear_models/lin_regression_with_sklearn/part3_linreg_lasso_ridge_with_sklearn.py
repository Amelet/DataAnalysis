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
- linear regression (does not handle well collinearity in matrix)
- lasso regression (Lasso is better for penalizing coefficients for uninformative features)
- ridge regression

# Ridge and Lasso regression are some of the simple techniques to reduce model complexity
# and prevent over-fitting which may result from simple linear regression.")
# info from https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

#"Ridge Regression : In ridge regression, the cost function is altered by adding a penalty equivalent
# to square of the magnitude of the coefficients.
# So ridge regression puts constraint on the coefficients (w). The penalty term (lambda) regularizes
# the coefficients such that if the coefficients take large values the optimization function is penalized.
# So, ridge regression shrinks the coefficients and it helps to reduce the model complexity and multi-collinearity.
# When λ → 0 , the cost function becomes similar to the linear regression cost function.
# So lower the constraint (low λ) on the features, the model will resemble linear regression model.

# Lasso cost function is just like Ridge regression cost function, for lambda =0, 
# The only difference is instead of taking the square of the coefficients, magnitudes are taken into account""")


# LINEAR REGRESSION function:
def get_models_coefs(train_data, train_labels, linreg_type, regularization_coef):
    """Accepts a choice of linreg model (normal linreg, lasso, ridge)
    and chosen regularization coefficient (for lasso, ridge).
    Workflow:
    1) Creates linreg object.
    2) Fits model.
    3) Returns model's coefficients"""
    if linreg_type == "linreg":
        linear_regressor = LinearRegression()                       # 1. Linreg object is created
        linear_regressor.fit(train_data, train_labels)      # 2. Model training
        coefficients = linear_regressor.coef_                       # 3. Linreg coefficients obtained
        return coefficients, linear_regressor
    elif linreg_type == "lasso":
        lasso_regressor = Lasso(random_state=3, alpha=regularization_coef)
        lasso_regressor.fit(train_data, train_labels)
        coefficients = lasso_regressor.coef_
        return coefficients, lasso_regressor
    elif linreg_type == "ridge":
        ridge_regressor = Ridge(random_state=3, alpha=regularization_coef)
        ridge_regressor.fit(train_data, train_labels)
        coefficients = ridge_regressor.coef_
        return coefficients, ridge_regressor
    else:
        print("Regression model was not found")
        coefficients = []


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