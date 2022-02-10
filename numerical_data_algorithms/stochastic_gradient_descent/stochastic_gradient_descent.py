import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def predict_y(X, w_coefs):
    """y_predicted from the linreg model"""
    return X.dot(w_coefs)


def mserror(y, y_pred):
    """MSE (mean square error)"""
    return np.sum((y - y_pred)**2)/len(y)


def stochastic_gradient_step(X, y, w_coefs, train_ind, eta=0.01):
    l, num_of_vars = X.shape[0], X.shape[1]
    k = 2 / l
    X_trained = X[train_ind]
    y_trained = y[train_ind]
    grad_by_var = []
    for i in range(0, num_of_vars, 1):
        grad = k * X_trained[i] * (predict_y(X_trained, w_coefs) - y_trained)
        grad_by_var.append(grad)
    return w_coefs - eta * np.array(grad_by_var)


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # initialize a large distance between w_coefs vectors
    weight_dist = np.inf
    # initialize w_coef
    w = w_init
    # errors are recorded at each iteration
    errors = []

    # iterations counter
    iter_num = 0

    # for random sampling
    np.random.seed(seed)

    while weight_dist > min_weight_dist and iter_num < max_iter:
        # sample random rows of X matrix
        random_ind = np.random.randint(X.shape[0])

        # do stochastic gradient step -> correct w_coefs at this iteration
        w_iter = stochastic_gradient_step(X, y, w, random_ind, eta=eta)

        # y_predicted
        y_pred = predict_y(X, w_iter)

        # note MSE at the iteration and save it
        error = np.mean((y - y_pred) ** 2)
        errors = np.append(errors, error)

        # update the distance between two vectors of w_coefs
        weight_dist = np.linalg.norm(w - w_iter)

        # update w_coefs
        w = w_iter
        iter_num += 1
    return w, errors


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
    clmn_w0 = np.reshape(np.ones(len(X)), (len(X), 1))
    X = np.hstack((clmn_w0, X))
    return X, y


# PART 1: DATA
adver_data = pd.read_csv('advertising.csv')
X, y = prepare_data_matrix(adver_data)

# PART 2: FIND COEFFICIENTS
w_init = np.zeros(X.shape[1])

stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e5,
                                                                            min_weight_dist=1e-8, seed=42,
                                                                            verbose=False)

# PART 3: visualizations
fig1 = plt.Figure()
plt.plot(stoch_errors_by_iter)
plt.xlabel("Iteration number")
plt.ylabel("MSE")
plt.title("Stochastic gradient descent convergence")
plt.show()



# Format data for printing results
coefs_list = np.hstack((["w0"], adver_data.columns.values[0:-1]))
w_coefs = stoch_grad_desc_weights
y_pred = predict_y(X, w_coefs)

print(f"""
Stochastic Gradient Descent is used to find coefficients for the linear regression.
Linreg's MSE = {mserror(y, y_pred)}
LinRegression model's coefficients:""")
for coef, coef_val in zip(coefs_list, w_coefs):
    print(f"\t{coef} = {round(coef_val, 2)}")