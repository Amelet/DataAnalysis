import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


# functions:
def split_train_test(X_real, X_cat, y, stratify):
    """Splits data into train test, and stratifies it as an option"""
    if stratify:
        (X_train_real,
         X_test_real, y_train, y_test) = train_test_split(X_real, y,
                                                          test_size=0.3,
                                                          random_state=0, stratify=y)
        (X_train_cat,
         X_test_cat) = train_test_split(X_cat,
                                        test_size=0.3,
                                        random_state=0, stratify=y)
        return X_train_real, X_test_real, X_train_cat, X_test_cat, y_train, y_test
    else:
        (X_train_real,
         X_test_real, y_train, y_test) = train_test_split(X_real, y,
                                                          test_size=0.3,
                                                          random_state=0)
        (X_train_cat,
         X_test_cat) = train_test_split(X_cat,
                                        test_size=0.3,
                                        random_state=0)
        return X_train_real, X_test_real, X_train_cat, X_test_cat, y_train, y_test


def transform_num_features(X_train_real, X_test_real, transform_features):
    """Transform numerical features using Polynomial = 2"""
    if transform_features:
        # transformation of numerical features
        # initialization
        transform = PolynomialFeatures(2)
        # apply transform to X train and X test
        X_train_real_poly = transform.fit_transform(X_train_real)
        X_test_real_poly = transform.transform(X_test_real)
        return X_train_real_poly, X_test_real_poly
    else:
        return X_train_real, X_test_real


# data scaling
def scale_data(x_train_real, x_test_real, x_train_categorical, x_test_categorical, y_train):
    """Scales features of X data"""
    scaler = StandardScaler()
    scaler.fit(x_train_real, y_train)
    X_train_real_scaled = scaler.transform(x_train_real)
    X_test_real_scaled = scaler.transform(x_test_real)
    # a) train data set
    X_scaled_train = np.hstack((X_train_real_scaled, x_train_categorical))
    # b) test data set
    X_scaled_test = np.hstack((X_test_real_scaled, x_test_categorical))
    return X_scaled_train, X_scaled_test


def sample_more_data(X_scaled_train, y_train):
    """Samples random rows from the underrepresented class"""
    np.random.seed(0)
    cl0 = len(y_train[y_train == 0])
    cl1 = len(y_train[y_train == 1])
    to_add = abs(cl0-cl1)
    if cl1 < cl0:
        smaller_class = 1
    else:
        smaller_class = 0
    print("\tsmaller class is: ", smaller_class, "\tneed to add ", to_add, "data points to train data set")
    indices_to_add = np.random.randint(0, np.sum(y_train == smaller_class)+1, to_add)

    X_subset = X_scaled_train[y_train.values == smaller_class]
    X_train_to_add = X_subset[indices_to_add, :]
    y_train_to_add = np.ones(to_add)
    # balance the data set with these data
    X_scaled_train_b = np.vstack((X_scaled_train, X_train_to_add))
    y_train_b = np.hstack((y_train, y_train_to_add))
    return X_scaled_train_b, y_train_b


def train_test_LRmodel(X_scaled_train, X_scaled_test, y_train, clweight, param_grid, cv):
    """Initializes LogisticRegression, does a search of a penalty coef for L2 regularization
    Then fits model to X_train/y_train, and predicts classes from X_test
    Uses ROC-AUC score to quantify model's quality"""
    if clweight != "None":
        regressor_z_scaled_b = LogisticRegression(random_state=0, solver='liblinear', class_weight=clweight)
    else:
        regressor_z_scaled_b = LogisticRegression(random_state=0, solver='liblinear')
    grid_cv_z_scaled_b = GridSearchCV(regressor_z_scaled_b, param_grid, cv=cv)
    grid_cv_z_scaled_b.fit(X_scaled_train, y_train)
    print("\tbest score: ", grid_cv_z_scaled_b.best_score_)
    print("\tbest parameters: ", grid_cv_z_scaled_b.best_params_)
    auc = roc_auc_score(y_test, grid_cv_z_scaled_b.predict_proba(X_scaled_test)[:, 1])
    print("\tROC-AUC metric: ", auc)


def plot_scores(optimizer):
    """Plots the best parameter after Grid Search"""
    param_C = [row['C'] for row in optimizer.cv_results_['params']]
    test_score = optimizer.cv_results_['mean_test_score']
    std_test_score = optimizer.cv_results_['std_test_score']
    plt.fill_between(par_C, test_score-std_test_score,
                     test_score+std_test_score, alpha=0.3)
    plt.semilogx(par_C, test_score)
    plt.show()


print("""
0. Load data""")
data = pd.read_csv('data.csv')
X = data.drop('Grant.Status', 1)
y = data['Grant.Status']
print("""
Evaluate how many are NaN values""")
print("\thow many rows in the data", data.shape)
print("\thow many non-NaN values in the data", data.dropna().shape) # we see, we cannot drop them, too many NaNs



print("""
1. SPLIT DATA INTO NUMERIC AND CATEGORICAL:
    1.1 find numeric columns and categorical columns""")
numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))


print("""
    1.2. split data into numerical and categorical data""")
X_real = X[numeric_cols].copy()
X_cat = X[categorical_cols].copy()

# 3.a. substitute numerical NaNs with zeros
# X_real_zeros = X_real.copy().fillna(0)


print("""
2. TREAT DIFFERENTLY NUMERICAL AND CATEGORICAL DATA:
    2.1. Numerical: substitute numerical NaNs with column's mean""")
X_real_means = X_real.copy()
means = X_real.mean()
for column in X_real.columns:
    print("for column '", column, "'   mean value = ", means[column])
    X_real_means[column] = X_real_means[column].fillna(means[column])


print("""
    2.2. Categorical: substitute categorical NaNs with 'NA'""")
X_cat.fillna('NA', inplace=True)
X_cat = X_cat.astype(str)


print("""
    2.3. Categorical: turn categorical data into numeric""")
encoder = DV(sparse=False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())
print("\tShape of categorical data", X_cat.shape)
print("\tShape of categorical data transformed to numerical", X_cat_oh.shape)


print("""
3. Splitting data into train/test""")
stratify = True
X_train_real, X_test_real, X_train_cat, X_test_cat, y_train, y_test = \
    split_train_test(X_real_means, X_cat_oh, y, stratify)
print("stratify is ", stratify)


print("""
4. Transformation of numerical features, use polynomial transformation.""")
transform_features = False
print("\tTransform numerical features with PolynomialFeautures? ", transform_features)
X_train_real, X_test_real = transform_num_features(X_train_real, X_test_real, transform_features)


print("""
5. From split data create two X data sets: X train (real+cathegorical), X test (real+categorical)""")
# 5.a. Train data for NaN zeros, NaN means
# X_rc_zeros = np.hstack((X_train_real_zeros, X_train_cat_oh))
X_rc_mean = np.hstack((X_train_real, X_train_cat))

# 5.b. Test data for NaN zeros, NaN means
# X_test_zeros = np.hstack((X_test_real_zeros, X_test_cat_oh))
X_test_mean = np.hstack((X_test_real, X_test_cat))

print("""
6. Grid Search and fit LR model""")
# 6.a. Classifier and its parameters
# penalty_is = "l1"    # (LASSO regression)
penalty_is = "l2"  # (Ridge regression)
regressor_m = LogisticRegression(random_state=0, solver='liblinear', penalty=penalty_is)
regressor_m.get_params()
print("\tPenalty for LogRegression is ", penalty_is)
# 6.b. Grid search
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}   # regularization strength for grid search
cv = 3
grid_cv_m = GridSearchCV(regressor_m, param_grid, cv=3)
# 6.c. Fit
grid_cv_m.fit(X_rc_mean, y_train)
print("\tbest score: ", grid_cv_m.best_score_)
print("\tbest parameters", grid_cv_m.best_params_)


print("""
7. Predict y from X data set (cl_m), predict probabilities (pp_m), measure ROC-AUC on predictions from X test""")
cl_m = grid_cv_m.predict(X_rc_mean)
pp_m = grid_cv_m.predict_proba(X_rc_mean)
auc_unbalanced = roc_auc_score(y_test, grid_cv_m.predict_proba(X_test_mean)[:,1])
print("\tFor unbalanced data set:")
print('\tROC-AUC measured X_test_mean data set', auc_unbalanced)


print("""
8. Cross validation score""")
scoring_m = cross_val_score(regressor_m, X_rc_mean, y_train, scoring='neg_mean_absolute_error', cv=3)
print('\tCross-Val score\nmean: {}, std: {}'.format(scoring_m.mean(), scoring_m.std()))

print("""
9. Plot: grid search for the best parameter C""")
scores_mean_m = grid_cv_m.cv_results_['mean_test_score']
scores_std_m = grid_cv_m.cv_results_['std_test_score']
C_param = param_grid['C']
plt.plot(C_param, scores_mean_m, '-*')
plt.fill_between(C_param, scores_mean_m+scores_std_m,scores_mean_m-scores_std_m, alpha= .1)
plt.title("Scores mean and std as a function of C parameter of grid search")
plt.xlabel("C parameter")
plt.ylabel("score")
plt.show()


# SUPPLEMENTARY:
# Data Preprocessing strategies
# SCALING numerical data and CLASS balancing
# a)
print(f"""
10. CLASS BALANCING: Classes are unbalanced;
\tnumber of entries for class 0 is {np.sum(y_train==0)}
\tnumber of entries for class 1 is {np.sum(y_train==1)}

BALANCING via class_weight in LogisticRegression
\t10.1.1) Scala data first --> def scale_data()
\t10.1.2) Use class-weight BALANCED as one of ways to train model""")
# Set a weight for classes (example: add class_weight to Logistic Regression):
X_scaled_train, X_scaled_test = scale_data(X_train_real, X_test_real, X_train_cat, X_test_cat, y_train)
clweight = "balanced"
train_test_LRmodel(X_scaled_train, X_scaled_test, y_train, clweight, param_grid, cv)

# b)
print("""
BALANCING via sampling from underrepresented class
\t10.2.1) Scale data first --> def scale_data
\t10.2.2) balancing classes via random sampling from underrepresented call""")
X_scaled_train_b, y_train_b = sample_more_data(X_scaled_train, y_train)
print(f"\trows are added, now class 0 has {np.sum(y_train_b == 0)}, class 1 has {np.sum(y_train_b == 1)}")
clweight = "None"
train_test_LRmodel(X_scaled_train_b, X_scaled_test, y_train_b, clweight, param_grid, cv)