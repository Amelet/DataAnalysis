import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# find a min of a function:
print("""
Find local minimum of the function:
    np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
Method: BFGS.
User needs to specify initial approximation of a minimum, its x coordinate""")


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


while True:
    try:
        users_input = input("Type initial approximation of the function's minimum in the range 1-32: ")
        users_input = float(users_input)
        break
    except ValueError:
        print("Please, type a single digit")

res = minimize(f, (users_input), method="BFGS")
print(f"Minimum is found at coordinates x = {np.round(res.x[0], 2)}, y = {np.round(f(res.x[0]), 2)}")
x = np.arange(1, 33, 1)


fig = plt.figure()
plt.scatter(users_input, f(users_input), color="b", label="initial approximation given by user")
plt.scatter(res.x, f(res.x), color="k", label="BFGS result: found minimum of the function")
plt.plot(x, f(x), label="f(x): np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)")
plt.legend(loc='lower left', prop={'size': 8})
plt.title("Search for function's minimum with BFGS")
plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()
