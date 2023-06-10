## 📂Numerical data (algorithms)
### 📁Linear models
#### 📁Least squares method
🐍least_squares_method.py
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

#### 📁Normal equation
🐍normal_equation.py
This Python script demonstrates how to build a simple linear regression model using the normal equation method. It calculates the coefficients for a linear model that predicts the dependent variable from the independent variables in an advertising dataset.

Here are the steps of the script:

1. **Define helper functions**: The script first defines several helper functions:

   - `normal_equation(X, y)`: Computes the coefficients of the linear regression using the normal equation method.
   - `predict_y(X, w_coefs)`: Predicts the target variable `y` using the design matrix `X` and the coefficients `w_coefs`.
   - `mserror(y, y_pred)`: Computes the mean square error between the actual target variable `y` and its prediction `y_pred`.
   - `scale_data(X)`: Scales the data to have zero mean and unit standard deviation.
   - `prepare_data_matrix(adver_data)`: Prepares the design matrix for the normal equation method.

2. **Load the data**: The script loads the advertising dataset using the Pandas `read_csv` function.

3. **Prepare the data matrix**: It uses the `prepare_data_matrix` function to prepare the design matrix `X` and the target variable vector `y`.

4. **Calculate coefficients**: It uses the `normal_equation` function to calculate the coefficients of the linear regression.

5. **Predict the target variable**: It uses the `predict_y` function to predict the target variable.

6. **Compute and display the mean square error**: It uses the `mserror` function to compute the mean square error between the actual and predicted target variable. It then prints the coefficients of the linear regression model and the mean square error.

The script is specifically designed for the `advertising.csv` dataset, which must be formatted in a specific way for the script to work. The dependent variable must be the last column, and all independent variables must be the preceding columns. All data must be numerical. The script scales all the independent variables before applying the normal equation method. The coefficients for the linear regression are then printed out, alongside the mean squared error of the model predictions.

#### 📁Linear Classification
🐍SGDC_with_grid_search.py
This project showcases how to use grid search to find the optimal parameters for a classifier in Scikit-learn. The Iris dataset is used as an example, with the SGDClassifier as the classifier of choice. The project begins by initializing the classifier and exploring its parameters to find the ones that will be searched for optimal values. A cross-validation object is then created for the grid search. The project then proceeds with the grid search, with the option to perform a random grid search or a regular grid search. The results of the grid search are displayed, including the best estimator, best score, and best parameters.

🐍part1_lin_classify_sklearn.py
This is a Python script for linear classification using the scikit-learn library. The script allows the user to choose between two linear classification models: logistic regression and ridge regression. The script generates a sample dataset using scikit-learn's make_blobs function, and splits it into training and test sets using scikit-learn's train_test_split function. The script then trains the selected linear classifier on the training set and evaluates its performance using accuracy scores obtained through two cross-validation strategies: 10-fold cross-validation and a stratified shuffle split cross-validation. Finally, the script prints the mean, max, min, and standard deviation of the accuracy scores obtained through each cross-validation strategy.

🐍part2_metrics.py
This Python script is used for evaluating the quality of different classification algorithms by analyzing various performance metrics. Specifically, it utilizes various metrics from Scikit-Learn library to compute the precision, recall, accuracy, F1 score, log loss, and a custom weighted log loss. Then, it visualizes these results.

The script operates in the following steps:

1. **LOAD DATA**: The script loads CSV files, each representing predictions from different classification algorithms. Each line in the CSV file has two columns: the first column for the actual label and the second for the predicted probability.

2. **SET THRESHOLD**: The threshold is initially set at 0.5. This is used to decide whether the predicted probability indicates class 1 or 0. If the predicted probability is greater than the threshold, it is considered as class 1, otherwise class 0.

3. **COMPUTE METRICS**: It computes several metrics (precision, recall, accuracy, f1 score, log loss, and a custom weighted log loss) using the actual and predicted classes. These metrics provide a quantitative measure of the performance of the classification algorithms.

4. **FIND OPTIMAL THRESHOLD**: This section uses the function `find_t` to compute the F1 scores for a range of thresholds (from 0.1 to 1.0). The function also computes the precision-recall curve. 

5. **PLOT RESULTS**: Finally, it visualizes the results with two subplots. The first subplot shows the precision-recall curve as a function of threshold, and the second subplot shows the F1 score as a function of threshold, highlighting the threshold that yields the maximum F1 score. The plots are shown for each algorithm, providing a graphical comparison of their performances.

The script is repeated for several different algorithms ('Ideal', 'Typical', 'Awful', 'Avoids FP', 'Avoids FN'), each with their own set of predicted probabilities. The purpose is to compare these algorithms in terms of the aforementioned metrics and their precision-recall and F1 score curves.


### 📁Data preprocessing strategies
🐍data_preprocessing.py
This project focuses on predicting grant approval by a research funding organization using logistic regression.

The data used in the project contains information about the grant applicants such as their demographics, research fields, and the number of successful and unsuccessful grants in the past.

The project involves several data preprocessing techniques such as handling missing values, transforming categorical data into numerical data, and scaling the numerical data. The project also includes strategies for balancing the unbalanced classes in the dataset such as using class weight balanced and oversampling from the underrepresented class.

The main objective of the project is to fit a logistic regression model to the preprocessed data and predict the grant approval for new applications. Grid search is used to find the optimal value of the regularization parameter C, and the quality of the model is evaluated using the ROC-AUC score. Cross-validation is also used to estimate the performance of the model. Finally, the results are visualized using plots.

### 📁Cross-validation strategies
This folder contains code snippets for ways to cross-validate data. The main purpose of this code is to demonstrate how to split arrays or matrices into random train and test subsets, and to show how to perform different types of cross-validation using the scikit-learn library.

The following functions are provided:

`train_test_split(data, target)`: splits the input data and target arrays or matrices into random train and test subsets using the train_test_split function from scikit-learn.<br>
`get_KFold()`: splits the dataset into k consecutive folds (without shuffling by default) using the KFold function from scikit-learn.<br>
`get_stratified_KFold(X, target)`: returns stratified folds using the StratifiedKFold function from scikit-learn. The folds are made by preserving the percentage of samples for each class.<br>
`get_shuffle_split(X)`: splits the input data into random subsets using the ShuffleSplit function from scikit-learn.<br>
`get_leave_one_out()`: performs leave-one-out cross-validation using the LeaveOneOut function from scikit-learn. Each sample is used once as a test set while the remaining samples form the training set.<br>
