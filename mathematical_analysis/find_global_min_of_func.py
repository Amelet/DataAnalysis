import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# find a min of a function:
print("""
Find global minimum of the function:
    np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
Method: differential_evolution.
User needs to specify the range of x-coordinates for the search""")


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


while True:
    try:
        users_input = input("\n\nType start/end of the range of x-coordinates to search for the function's minimum:"
                            "\n\tformat: x_min x_max"
                            "\n\tfor example: 1 31"
                            "\nType the range:   ")
        users_input = users_input.split(sep=" ")
        users_input = tuple(int(x) for x in users_input)
        break
    except ValueError:
        print("Please, type two digits separated by space")
print("Your range: ", users_input)
res = differential_evolution(f, [users_input])

print(f"Minimum is found at coordinates x = {np.round(res.x[0], 2)}, y = {np.round(f(res.x[0]), 2)}")
x = np.arange(users_input[0], users_input[1], 1)


fig = plt.figure()
plt.scatter(res.x, f(res.x), color="k", label="differential_evaluation result: found minimum of the function")
plt.plot(x, f(x), label="f(x): np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)")
plt.legend(loc='lower left', prop={'size': 8})
plt.title("Search for function's minimum with differential evaluation")
plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()
