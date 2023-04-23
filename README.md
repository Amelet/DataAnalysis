# DataAnalysis

Welcome to my Data Analysis repository! Here you can find a collection of mini-projects that I have completed to analyze data using Python. Each project is contained in its own .py file and includes specific functions and a data set on which the functions are tested.

## Numerical data

### _**Central Limit Theorem:**_

This project demonstrates how the Central Limit Theorem can be implemented to estimate the mean of a distribution. As an example, I take the chi-squared distribution and use it to estimate its mean by:

Drawing 1000 samples from the chi-squared distribution, where each sample is an arithmetic mean of N random observations. N is set by the user.
Fitting a Gaussian distribution to each of the 1000 sample distributions.
Estimating the mean of the chi-squared distribution from the Gaussian distribution.
Comparing the estimated mean with the true mean of the chi-squared distribution.

### _**Find global min of a function:**_

    np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
Using method: differential_evolution.
User specifies the x-range for a minimum to be found


### _**Find min of a function:**_

    np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
Using method: BFGS.
User specifies the initial aproximation of the minimum x-coordinate


### _**LinAlg Solve:**_

Task: Fit n-degree polynomial to a function:

      np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
User is going to specify several x-coordinates for a polynomial to pass

## Text data
### _**cosine distance between sentences**_
This script takes a .txt file of sentences and measures how similar every of them to the first sentence.
Similarity measure is cosine distance.

The algorithm's output: the index of the closest sentence and its cosine distance to the first sentence.
If words change the form, it is not accounted. A naive way to compare sentences
