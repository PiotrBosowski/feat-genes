import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from model_committee.data_preparation import mask_columns
from model_committee.plotting import valid_test_acc_curves
from model_committee.results import get_metrics, get_avg_metrics


def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores,
                                                            np.mean(scores),
                                                            np.std(scores)))


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def train_xgboost(train_X, train_y, valid_X, valid_y, test_X, test_y,
                  learning_rate=None, mask=None):
    learning_rate = 0.01
    n_estimators = 350
    subsample = 0.3
    if mask:
        train_X = mask_columns(train_X, mask)
        valid_X = mask_columns(valid_X, mask)
        test_X = mask_columns(test_X, mask)
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                                  tree_method='gpu_hist',
                                  predictor='gpu_predictor')
    xgb_model.learning_rate = learning_rate
    xgb_model.n_estimators = n_estimators
    xgb_model.subsample = subsample
    xgb_model.fit(train_X, train_y.values.ravel())
    train_pred = xgb_model.predict(train_X)
    valid_pred = xgb_model.predict(valid_X)
    test_pred = xgb_model.predict(test_X)
    return (get_metrics(train_y, train_pred, verbose=True),
            get_metrics(valid_y, valid_pred, verbose=True),
            get_metrics(test_y, test_pred, verbose=True))


def train_xgboost_gridsearch(train_X, train_y, valid_X, valid_y, test_X,
                             test_y,
                             learning_rate=None, mask=None):
    if mask:
        train_X = mask_columns(train_X, mask)
        valid_X = mask_columns(valid_X, mask)
        test_X = mask_columns(test_X, mask)

    param_grid_gb = {'learning_rate': [0.01],
                     'n_estimators': [350],
                     'subsample': [0.3]}

    # Regressor Instantiation
    gb = xgb.XGBClassifier()

    mse_grid = GridSearchCV(estimator=gb, param_grid=param_grid_gb,
                            scoring='neg_mean_squared_error', cv=4,
                            verbose=2)
    # xgb_model = xgb.XGBClassifier(objective="binary:logistic")
    mse_grid.fit(train_X, train_y.values.ravel())
    train_pred = mse_grid.predict(train_X)
    valid_pred = mse_grid.predict(valid_X)
    test_pred = mse_grid.predict(test_X)
    print("Best parameters:", mse_grid.best_params_)
    return (get_metrics(train_y, train_pred, verbose=True),
            get_metrics(valid_y, valid_pred, verbose=True),
            get_metrics(test_y, test_pred, verbose=True))


def train_xgboost_repeatedly(train_X, train_y, valid_X, valid_y, test_X,
                             test_y,
                             masks_list=None):
    metrics_bundles = []
    for mask in masks_list:
        metrics_bundles.append(train_xgboost(train_X, train_y,
                                             valid_X, valid_y,
                                             test_X, test_y,
                                             mask=mask))
    train_metrics, valid_metrics, test_metrics = zip(*metrics_bundles)
    valid_test_accs = zip(list(valid_metrics), list(test_metrics))
    valid_test_acc_curves(valid_test_accs)
    get_avg_metrics(train_metrics)
    get_avg_metrics(valid_metrics)
    get_avg_metrics(test_metrics)
