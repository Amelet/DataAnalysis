from sklearn import model_selection, datasets


print("""
Code snippets for ways to cross-validate data""")


def train_test_split(data, target):
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target,
                                                                                        test_size=0.3)
    return train_data, test_data, train_labels, test_labels


def get_KFold():
    # kf = model_selection.KFold(n_splits=5)
    # kf = model_selection.KFold(n_splits=2, shuffle=True)
    kf = model_selection.KFold(n_splits=2, shuffle=True, random_state=1)
    return kf.split(X)


def get_stratified_KFold(X, target):
    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    return skf.split(X, target)


def get_suffle_split(X):
    ss = model_selection.ShuffleSplit(n_splits=10, test_size=0.2)
    return ss.split(X)


def get_leave_one_out():
    loo = model_selection.LeaveOneOut()
    return loo.split(X)