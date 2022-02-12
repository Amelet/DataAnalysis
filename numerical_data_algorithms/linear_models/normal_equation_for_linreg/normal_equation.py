import numpy as np
import pandas as pd


def normal_equation(X, y):
    """Returns coefficients of linear regression"""
    pseaudo_inv = np.linalg.pinv(X)
    w_coefs = pseaudo_inv.dot(y)
    return w_coefs


def predict_y(X, w_coefs):
    """y_predicted from the linreg model"""
    return X.dot(w_coefs)


def mserror(y, y_pred):
    """MSE (mean square error)"""
    return np.sum((y - y_pred)**2)/len(y)


def scale_data(X):
    """Data scaling to center it at mean = 0, and to have std = 1"""
    means, stds = np.mean(X, axis=0), np.std(X, axis=0)
    return (X - means) / stds


def prepare_data_matrix(adver_data):
    """Prepares X data matrix for normal equation.
    Data scaling is done in place"""
    clmns = adver_data.columns.values
    clmns_for_X = clmns[0:-1]
    clmns_for_y = clmns[-1]
    X = adver_data[clmns_for_X].values
    y = adver_data[clmns_for_y].values
    X = scale_data(X)
    # add a column of ones to fit w0 coef:
    clmn_w0 = np.reshape(np.ones(len(X)), (len(X),1))
    X = np.hstack((clmn_w0, X))
    return X, y


# PART 1: DATA
adver_data = pd.read_csv('advertising.csv')
X, y = prepare_data_matrix(adver_data)


# Get optimal coefficients for the linear model
w_coefs = normal_equation(X, y)

# Use model to predict y
y_pred = predict_y(X, w_coefs)

# Format data for printing results
coefs_list = np.hstack((["w0"], adver_data.columns.values[0:-1]))
print(f"""
Normal equation is used to find coefficients for the linear regression.
Linreg's MSE = {mserror(y, y_pred)}
LinRegression model's coefficients:""")
for coef, coef_val in zip(coefs_list, w_coefs):
    print(f"\t{coef} = {round(coef_val, 2)}")