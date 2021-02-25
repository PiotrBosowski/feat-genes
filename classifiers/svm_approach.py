from sklearn import svm

from utils.data_preparation import mask_columns, prepare_data
from utils.plotting import valid_test_acc_curves
from utils.results import get_metrics, get_avg_metrics


def train_svm(kernel, train_X, train_y, valid_X, valid_y, test_X, test_y, mask=None):
    if mask:
        train_X = mask_columns(train_X, mask)
        valid_X = mask_columns(valid_X, mask)
        test_X = mask_columns(test_X, mask)
    classifier = svm.SVC(kernel=kernel)
    classifier.fit(train_X, train_y.values.ravel())
    train_pred = classifier.predict(train_X)
    valid_pred = classifier.predict(valid_X)
    test_pred = classifier.predict(test_X)
    return (get_metrics(train_y, train_pred, verbose=True),
            get_metrics(valid_y, valid_pred, verbose=True),
            get_metrics(test_y, test_pred, verbose=True))


def train_svm_grid(train_path, test_path):
    train_X, train_y = prepare_data(train_path)
    test_X, test_y = prepare_data(test_path)
    for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
        for class_weight in [None, 'balanced']:
            for gamma in ['auto', 'scale']:
                for C in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
                    classifier = svm.SVC(kernel=kernel,
                                         class_weight=class_weight,
                                         gamma=gamma,
                                         C=C)
                    classifier.fit(train_X, train_y.values.ravel())
                    test_pred = classifier.predict(test_X)
                    train_pred = classifier.predict(train_X)
                    get_metrics(test_y, test_pred)
                    get_metrics(train_y, train_pred)


def train_svm_repeatedly(kernel, train_X, train_y, valid_X, valid_y, test_X,
                             test_y,
                             masks_list=None):
    metrics_bundles = []
    for mask in masks_list:
        metrics_bundles.append(train_svm(kernel, train_X, train_y,
                                             valid_X, valid_y,
                                             test_X, test_y,
                                             mask=mask))
    train_metrics, valid_metrics, test_metrics = zip(*metrics_bundles)
    valid_test_accs = zip(list(valid_metrics), list(test_metrics))
    valid_test_acc_curves(valid_test_accs)
    get_avg_metrics(train_metrics)
    get_avg_metrics(valid_metrics)
    get_avg_metrics(test_metrics)
