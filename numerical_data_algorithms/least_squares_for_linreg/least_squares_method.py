import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def linear_model(w0, w1, x1):
    """returns linear model: w0 + w1*x1"""
    return w0 + w1 * x1


def quadratic_error(params, x, y, for_fit):
    """returns quadratic error between y and y_predicted"""
    w0, w1 = (params[0], params[1])
    err = (y - linear_model(w0, w1, x)) ** 2
    if not for_fit:
        return err
    else:
        return np.sum(err)


file_name = 'weights_heights.csv'

# PART 1: DATA
# load dataset
data = pd.read_csv(file_name, index_col='Index')

# show pairwise dependency between data features
g = sns.pairplot(data)
plt.suptitle(f"Pairplot for the data set: {file_name}", x=0.6, y=0.99)
plt.show()

# we will fit linear model to Height vs Weight dependency
# retrieve x,y data
x_range = data["Weight"]
y_range = data["Height"]


# PART II: SNIPPET FOR LEAST SQUARES METHOD
# minimize quadratic error: find optimal w0, w1
res = minimize(quadratic_error, x0=[0, 0], args=(x_range, y_range, True), method="L-BFGS-B",
               bounds=[(-100, 100), (-5, 5)])
w0_opt = res.x[0]
w1_opt = res.x[1]
q_err_opt = quadratic_error([res.x[0], res.x[1]], x_range, y_range, True)

# visualize the linear model of height dependency on weight
fig2 = plt.figure()
plt.scatter(x_range, y_range, label="experimental data", alpha=0.5)
label = "w0 = " + str(np.round(w0_opt, 2)) + "\nw1 = " + str(np.round(w1_opt,2))
plt.plot(x_range, linear_model(w0_opt, w1_opt, x_range), 'r', label = label)
plt.legend()
plt.xlabel("Weight, pounds")
plt.ylabel("Height, inches")
plt.title("Linear model of dependency between people's height and weight")
plt.show()


# SUPPLEMENTARY
# visualize how quadratic error depends on w0, w1 and optimal (w0, w1)
# prepare to visualize quadratic error dependency on w0, w1
w0_range = np.arange(40, 60, 0.25)
w1_range = np.arange(-5, 5, 0.25)
w0_range, w1_range = np.meshgrid(w0_range, w1_range)
q_error = np.array([quadratic_error([w0, w1], x, y, False) for y, w0, w1, x in
                    zip(x_range, np.ravel(w0_range), np.ravel(w1_range), y_range)])
q_error = q_error.reshape(w0_range.shape)

fig3 = plt.figure(figsize=[15, 15])
ax = fig3.gca(projection='3d')
surf = ax.plot_surface(w0_range, w1_range, q_error, alpha=0.3)
ax.set_xlabel('w0, intercept')
ax.set_ylabel('w1, slope')
ax.set_zlabel('error')
ax.set_title(f"Quadratic error dependency on coef w0, w1\noptimal w0 = {round(w0_opt, 2)}, w1 = {round(w1_opt, 2)}, "
             f"q_err = {round(q_err_opt, 2)}")
ax.scatter(w0_opt, w1_opt, q_err_opt, marker='o', color="r")
plt.show()