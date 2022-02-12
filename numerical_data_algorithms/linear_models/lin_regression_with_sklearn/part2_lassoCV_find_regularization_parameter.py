from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
np.set_printoptions(precision=3, suppress=True)


def lasso_regression_optimize(X, y, coef_range):
    lassocv_regressor = LassoCV(alphas=coef_range, cv=3, random_state=3)
    lassocv_regressor.fit(X, y)
    best_alpha = lassocv_regressor.alpha_
    coefficients = lassocv_regressor.coef_
    mse_vs_alpha = np.mean(lassocv_regressor.mse_path_, axis=1)
    alphas = lassocv_regressor.alphas_
    return best_alpha, coefficients, mse_vs_alpha, alphas


# CHOOSE RANGE of regularization parameter to assess the best value for the optimum MSE of the model
def choose_coef_range():
    while True:
        try:
            coef_range = input("""
    Please choose the range for the regularization coefficient.
        For example: 1,50,5
    This will be transformed into the range from 1 to 50 with increment = 5
    Type your range parameters here:    """)
            coef_range = coef_range.split(sep=",")
            start = float(coef_range[0])
            end = float(coef_range[1])
            step = float(coef_range[2])
            coef_range = np.arange(start, end, step)
            break
        except ValueError:
            print("Pleae, type three digits, they should be COMMA separated")
    return coef_range


# PREPARE DATA (load, shuffle, scale it)
df = pd.read_csv("bikes_rent.csv")
df_shuffled = shuffle(df, random_state=123)
X = scale(df_shuffled[df_shuffled.columns[:-1]])
y = df_shuffled.iloc[:, -1]


# GET USERs CHOICE for the regularization coefficient range to find the optimal coefficient value
coef_range = choose_coef_range()


# TRAIN MODEL
best_alpha, coefficients, mse_vs_alpha, alphas = lasso_regression_optimize(X, y, coef_range)


# PLOT OPTIMIZATION of ALPHA regularization coefficient
fig1 = plt.figure()
plt.plot(alphas, mse_vs_alpha, label="MSE path")
plt.scatter(best_alpha, np.min(mse_vs_alpha), color="r", label=f"Best alpha {best_alpha}")
plt.legend()
plt.title("MSE dependency on LASSO's regularization coefficient (alpha)")
plt.xlabel("alpha")
plt.ylabel("MSE of LASSO regression model")
plt.tight_layout()
plt.show()


# PRINT results
try:
    if len(coefficients)>0:
        print("Model's coefficients:")
        for variable, coef in zip(df.columns, coefficients):
            print(f"\t{variable}, {coef}")
except:
    print("Coefficients were not calculated")