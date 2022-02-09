import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

print("Task: Fit n-degree polynomial to a function:"
      "\n\tnp.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)"
      "\nUser is going to specify several x-coordinates for a polynomial to pass")


# given function f(x)
def f(x):
    """A function to be fitted"""
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


def polyfit(x_coordinates):
    """Polynomial fit of the function"""
    polinomial_degree = len(x_coordinates) - 1
    # generate y
    y = [f(xx) for xx in x_coordinates]
    # create a matrix
    A = P.polyvander(x_coordinates, polinomial_degree)
    # find coeff w0...wn
    # 1*w0 + x1*w1 + ... xn*wn
    coefs = linalg.solve(A, y)
    print('coefficients are solved: ', np.round(coefs, 2))
    # function of the polynomial:
    f1 = P.Polynomial(coefs)
    return f1

while True:
    try:
        users_input = input("Give a list of x coordinates separated by space"
                      "\nin the range 1-15:  ")
        users_input = users_input.split(sep=" ")
        users_input = [float(x) for x in users_input]
        print("Your input: ", users_input)
        break
    except ValueError:
        print("Please, type only digits in the range 1-15 with separation  by space")
        continue


x = np.arange(1,16,1)
f1 = polyfit(users_input)

plt.figure()
plt.plot(x, f(x), label="np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)")
plt.scatter(users_input, [f(x_val) for x_val in users_input], color="k", label="(x, y) given by user")
plt.plot(x, f1(x), label="fitted polynomial of degree "+str(len(users_input)-1))
plt.title("Function approximation with a polynomial")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()