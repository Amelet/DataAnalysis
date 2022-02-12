from sklearn import model_selection, datasets


print("""
Code snippets for ways to cross-validate data""")


def train_test_split(data, target):
    """Split arrays or matrices into random train and test subsets."""
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target,
                                                                                        test_size=0.3)
    return train_data, test_data, train_labels, test_labels


def get_KFold():
    """Split dataset into k consecutive folds (without shuffling by default)"""
    # kf = model_selection.KFold(n_splits=5)
    # kf = model_selection.KFold(n_splits=2, shuffle=True)
    kf = model_selection.KFold(n_splits=2, shuffle=True, random_state=1)
    return kf.split(X)


def get_stratified_KFold(X, target):
    """This cross-validation object is a variation of KFold that returns stratified folds.
    The folds are made by preserving the percentage of samples for each class."""
    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    return skf.split(X, target)


def get_suffle_split(X):
    """Random permutation cross-validator"""
    ss = model_selection.ShuffleSplit(n_splits=10, test_size=0.2)
    return ss.split(X)


def get_leave_one_out():
    """Each sample is used once as a test set (singleton) while the remaining samples form the training set."""
    loo = model_selection.LeaveOneOut()
    return loo.split(X)