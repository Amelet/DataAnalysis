# DataAnalysis

Welcome to my Data Analysis repository! Here you can find a collection of mini-projects that I have completed to analyze data using Python. Each project is contained in its own .py file and includes specific functions and a data set on which the functions are tested.

## 📁Numerical data

### 🔢 _**Central Limit Theorem:**_

This project demonstrates how the Central Limit Theorem can be implemented to estimate the mean of a distribution. As an example, I take the chi-squared distribution and use it to estimate its mean by:

Drawing 1000 samples from the chi-squared distribution, where each sample is an arithmetic mean of N random observations. N is set by the user.
Fitting a Gaussian distribution to each of the 1000 sample distributions.
Estimating the mean of the chi-squared distribution from the Gaussian distribution.
Comparing the estimated mean with the true mean of the chi-squared distribution.

### 📈 _**Find global min of a function:**_
This project uses the differential_evolution method to find the global minimum of the function:

    np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
The user specifies the range of x-values in which to search for the minimum.


### 📈 _**Find min of a function:**_
This project uses the BFGS method to find the minimum of the function:

    np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
The user specifies the initial approximation of the x-coordinate of the minimum.


### 📈 _**LinAlg Solve:**_
This project fits an n-degree polynomial to the function:

      np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
The user specifies several x-coordinates for the polynomial to pass through.

## 📁Text data
### 📖_**cosine distance between sentences**_
This project measures the cosine distance between a collection of sentences in a .txt file and the first sentence in the file. The algorithm outputs the index of the closest sentence and its cosine distance to the first sentence. If words change form, they are not accounted for, and therefore a naive comparison of the sentences is used.
