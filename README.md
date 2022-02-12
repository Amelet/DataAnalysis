# DataAnalysis

In this repository, I collect scripts that I have used in data analysis. Each .py file organized as a mini project; it contains specific functions and a data set on which functions are tested.
## Numerical data

### _**Central Limit Theorem:**_

Shows how CLT can be implemented to estimate a mean of a distribution. As an example of a distribution, I take chi2.
To estimate its mean:
1. I draw 1000 of samples from chi2: each sample is a arithmetic mean of N random observations. N is set by the user.
2. I fit Gaussian distribution to a 1000 of samples distributions;
3. From Gaussian distribution I estimate the mean of chi2
4. Compare estimated mean with the mean of chi2.


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
