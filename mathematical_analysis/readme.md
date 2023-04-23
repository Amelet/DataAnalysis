# Mathematical analysis
## _**Central Limit Theorem:**_
🐍 central_limit_theorem.py<br>
This project demonstrates how the Central Limit Theorem can be implemented to estimate the mean of a distribution. As an example, I take the chi-squared distribution and use it to estimate its mean by:

Drawing 1000 samples from the chi-squared distribution, where each sample is an arithmetic mean of N random observations. N is set by the user.
Fitting a Gaussian distribution to each of the 1000 sample distributions.
Estimating the mean of the chi-squared distribution from the Gaussian distribution.
Comparing the estimated mean with the true mean of the chi-squared distribution.

## _**Find global min of a function:**_
🐍 find_global_min_of_func.py<br>
This project uses the differential_evolution method to find the global minimum of the function:<br>
`np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)`<br>
The user specifies the range of x-values in which to search for the minimum.


## _**Find min of a function:**_
🐍 find_min_of_func.py<br>
This project uses the BFGS method to find the minimum of the function:<br>
`np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)`<br>
The user specifies the initial approximation of the x-coordinate of the minimum.


## _**LinAlg Solve:**_
🐍 linalg_solve_function.py<br>
This project fits an n-degree polynomial to the function:<br>
`np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)`<br>
The user specifies several x-coordinates for the polynomial to pass through.
