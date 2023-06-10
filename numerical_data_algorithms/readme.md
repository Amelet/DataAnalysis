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

#### 📁Stochastic Gradient Descent Linear Regression
🐍stochastic_gradient_descent.py
This project provides a Python implementation of the Stochastic Gradient Descent (SGD) algorithm for Linear Regression. It involves data scaling, gradient calculation, error analysis, and visualization. It contains the following functions:

- `predict_y(X, w_coefs)`: This function predicts the output based on input features and weight coefficients.
- `mserror(y, y_pred)`: Computes the Mean Square Error (MSE) between actual and predicted outputs.
- `stochastic_gradient_step(X, y, w_coefs, train_ind, eta=0.01)`: Function to compute the SGD step.
- `stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4, min_weight_dist=1e-8, seed=42, verbose=False)`: Main function that performs SGD over a maximum number of iterations, updates weight coefficients, and computes MSE.
- `scale_data(X)`: Performs standardization on the input data.
- `prepare_data_matrix(adver_data)`: Prepares the data matrix, scales the features, and adds a column of ones to handle the bias term in the linear equation.

The project uses the `advertising.csv` dataset for performing linear regression using SGD. It then plots the convergence of the SGD algorithm by visualizing the decrease in MSE with each iteration.

The final output of the project is the optimized weight coefficients for the features, and the minimum MSE achieved.


#### 📁Linear Regression
🐍part1_data_evaluation_before_linregmodel.py
This project provides a set of Python functions designed to help in the initial stages of data analysis, particularly when preparing to fit a linear regression model. The functions cover tasks such as visualizing dependencies, checking correlation, and identifying collinearity and scale differences in a dataset.

Here are the key components of the project:

- `plot_pairwise_dependency(df)`: Visualizes the count of bike rentals as a function of various variables, using pairwise scatter plots. 

- `check_correlation_to_y(df)`: Calculates the Pearson correlation between the output column and each of the other columns in the dataset. This can help to identify which variables may be relevant for a linear regression model.

- `check_corr_pairwise(df, continuous_variables)`: Checks for collinearity between different columns in the dataset. If collinear columns are detected, regularization of the model might be necessary.

- `columns_means(df)`: Checks for scale differences between variables by returning the mean of each column. 

The project is intended to serve as a practical guide to the initial stages of data analysis, with a specific focus on preparing data for linear regression. While the code snippets are provided as functions, they also represent a potential workflow for analyzing a new dataset.

🐍part2_lassoCV_find_regularization_parameter.py
This project implements Lasso Regression, a type of linear regression that includes a regularization term to avoid overfitting, using sklearn's LassoCV model. The model iteratively fits the data along a regularization path to find the optimal alpha, the regularization parameter.

Key Functions:

- `lasso_regression_optimize(X, y, coef_range)`: Trains a LassoCV model with the given input data `X`, output data `y`, and range of regularization parameters `coef_range`. The function returns the optimal alpha value, the model's coefficients, and the mean squared error (MSE) for each alpha in the regularization path.

- `choose_coef_range()`: Prompts the user to input a range for the regularization coefficient, which will be used to find the optimal alpha value. The user must provide the start, end, and step size for the range.

The project includes a data preparation stage that loads, shuffles, and scales a bike rentals dataset. Then, it prompts the user to specify a range for the regularization coefficient. Using this range, the LassoCV model is trained to find the optimal alpha value. 

Finally, the optimization process is visualized by plotting the MSE against the alpha values. The resulting plot helps to demonstrate the performance of the Lasso Regression model as the regularization parameter changes.

Note: The code has been implemented in a way that informs the user when coefficients for the model were not calculated, ensuring transparency in model performance.

🐍part3_linreg_lasso_ridge_with_sklearn.py
This project implements and compares three types of linear regression models: Linear Regression, Lasso Regression, and Ridge Regression, using sklearn's library. It demonstrates how regularization techniques, Lasso and Ridge, help handle collinearity and reduce model overfitting.

Key Functions:

- `get_models_coefs(train_data, train_labels, linreg_type, regularization_coef)`: Trains a specified type of linear regression model (Linear, Lasso, Ridge) using given input data `train_data`, output data `train_labels`, and a regularization coefficient `regularization_coef` for Lasso and Ridge regression.

- `make_predictions(model, test_data, test_labels)`: Makes predictions using a trained model `model` and test data `test_data`. Returns the predictions and mean absolute error.

- `scoring_cross_validation(model, data, target)`: Performs cross-validation scoring on a trained model `model` using given data `data` and target `target`, returning the mean and standard deviation of the scores.

- `choose_linreg_type()` and `choose_regularization_coef()`: Prompts user to choose the type of linear regression model and regularization coefficient (for Lasso and Ridge) to use.

The project uses a bike rentals dataset, which is loaded, shuffled, and scaled. The user specifies the type of regression model and (if needed) the regularization coefficient. Then, the model is trained, and predictions are made on the test data. 

Additionally, cross-validation is performed to validate the model. The model's coefficients are printed out, and in case of any issues, the user is informed if coefficients were not calculated. 

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
