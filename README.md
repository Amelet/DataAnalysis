# DataAnalysis

Welcome to my Data Analysis repository! Here you can find a collection of mini-projects that I have completed to analyze data using Python. Each project is contained in its own .py file and includes specific functions and a data set on which the functions are tested.

## 📂Mathematical analysis
This folder contains four mini-projects for numerical data analysis.
### 🐍 _**Central Limit Theorem:**_

This project demonstrates how the Central Limit Theorem can be implemented to estimate the mean of a distribution. As an example, I take the chi-squared distribution and use it to estimate its mean by:

Drawing 1000 samples from the chi-squared distribution, where each sample is an arithmetic mean of N random observations. N is set by the user.
Fitting a Gaussian distribution to each of the 1000 sample distributions.
Estimating the mean of the chi-squared distribution from the Gaussian distribution.
Comparing the estimated mean with the true mean of the chi-squared distribution.

### 🐍 _**Find global min of a function:**_
This project uses the differential_evolution method to find the global minimum of the function:<br>
`np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)`<br>
The user specifies the range of x-values in which to search for the minimum.


### 🐍 _**Find min of a function:**_
This project uses the BFGS method to find the minimum of the function:<br>
`np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)`<br>
The user specifies the initial approximation of the x-coordinate of the minimum.


### 🐍 _**LinAlg Solve:**_
This project fits an n-degree polynomial to the function:<br>
`np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)`<br>
The user specifies several x-coordinates for the polynomial to pass through.

## 📂Numerical data (algorithms)
### 📁Linear models
#### 🐍Linear Regression: Fitting a Model to Height and Weight Data
This project demonstrates how to fit a linear model to a dataset of heights and weights (`weights_heights.csv` dataset is used).
The project uses Python's `NumPy`, `Pandas`, `Seaborn`, and `Matplotlib` libraries to:

Load and visualize the dataset using a pairwise dependency plot <br>
Implement the least squares method to find the optimal coefficients for the linear model <br>
Visualize the linear model of height dependency on weight <br>
Visualize how quadratic error depends on the optimal coefficients. <br>
The project uses the following functions:

`linear_model(w0, w1, x1)`: Returns a linear model w0 + w1*x1.<br>
`quadratic_error(params, x, y, for_fit)`: Returns the quadratic error between y and y_predicted.<br>
`train_test_split(data, target)`: Splits arrays or matrices into random train and test subsets.<br>
The `weights_heights.csv` dataset is used to demonstrate the implementation of these functions.

#### 📁Project: lin_classification_with_sklearn
🐍SGDC_with_grid_search.py
This project showcases how to use grid search to find the optimal parameters for a classifier in Scikit-learn. The Iris dataset is used as an example, with the SGDClassifier as the classifier of choice. The project begins by initializing the classifier and exploring its parameters to find the ones that will be searched for optimal values. A cross-validation object is then created for the grid search. The project then proceeds with the grid search, with the option to perform a random grid search or a regular grid search. The results of the grid search are displayed, including the best estimator, best score, and best parameters.

🐍part0_data_preprocessing.py
This project focuses on predicting grant approval by a research funding organization using logistic regression.

The data used in the project contains information about the grant applicants such as their demographics, research fields, and the number of successful and unsuccessful grants in the past.

The project involves several data preprocessing techniques such as handling missing values, transforming categorical data into numerical data, and scaling the numerical data. The project also includes strategies for balancing the unbalanced classes in the dataset such as using class weight balanced and oversampling from the underrepresented class.

The main objective of the project is to fit a logistic regression model to the preprocessed data and predict the grant approval for new applications. Grid search is used to find the optimal value of the regularization parameter C, and the quality of the model is evaluated using the ROC-AUC score. Cross-validation is also used to estimate the performance of the model. Finally, the results are visualized using plots.

🐍part1_data_preprocessing.py
This is a Python script for linear classification using the scikit-learn library. The script allows the user to choose between two linear classification models: logistic regression and ridge regression. The script generates a sample dataset using scikit-learn's make_blobs function, and splits it into training and test sets using scikit-learn's train_test_split function. The script then trains the selected linear classifier on the training set and evaluates its performance using accuracy scores obtained through two cross-validation strategies: 10-fold cross-validation and a stratified shuffle split cross-validation. Finally, the script prints the mean, max, min, and standard deviation of the accuracy scores obtained through each cross-validation strategy.

### 📁Cross-validation strategies
This folder contains code snippets for ways to cross-validate data. The main purpose of this code is to demonstrate how to split arrays or matrices into random train and test subsets, and to show how to perform different types of cross-validation using the scikit-learn library.

The following functions are provided:

`train_test_split(data, target)`: splits the input data and target arrays or matrices into random train and test subsets using the train_test_split function from scikit-learn.<br>
`get_KFold()`: splits the dataset into k consecutive folds (without shuffling by default) using the KFold function from scikit-learn.<br>
`get_stratified_KFold(X, target)`: returns stratified folds using the StratifiedKFold function from scikit-learn. The folds are made by preserving the percentage of samples for each class.<br>
`get_shuffle_split(X)`: splits the input data into random subsets using the ShuffleSplit function from scikit-learn.<br>
`get_leave_one_out()`: performs leave-one-out cross-validation using the LeaveOneOut function from scikit-learn. Each sample is used once as a test set while the remaining samples form the training set.<br>

## 📁Text data
### 🐍**cosine distance between sentences**
This project measures the cosine distance between a collection of sentences in a .txt file and the first sentence in the file. The algorithm outputs the index of the closest sentence and its cosine distance to the first sentence. If words change form, they are not accounted for, and therefore a naive comparison of the sentences is used.
