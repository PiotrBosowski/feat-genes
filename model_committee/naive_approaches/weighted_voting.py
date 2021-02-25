import pandas
import numpy as np
from sklearn import preprocessing

from model_committee.data_preparation import mask_columns
from model_committee.plotting import valid_test_acc_curves
from model_committee.results import get_metrics, get_avg_metrics

overview_path = '/home/peter/media/data/covid-19/models-BINARY-NEWEST/overview-2021-02-06_16-45-10.csv'

class WeightedVoting:
    def __init__(self, weights='f1score', threshold=0.5):
        metrics = pandas.read_csv(overview_path)
        arr = metrics[[weights]].values
        scaler = preprocessing.MinMaxScaler()
        self.weights = np.square(scaler.fit_transform(arr))
        self.threshold = threshold

    def predict(self, data_X, mask=None):
        if mask is None:
            mask = [True] * len(data_X.columns)
        indices = [ind for ind, item in enumerate(mask) if item]
        weights = [self.weights[ind] for ind in indices]
        for ind, col in enumerate(data_X.columns):
            weight = weights[ind]
            data_X[data_X.columns[ind]] *= weight
        votes = data_X.sum(axis=1).to_frame()
        weight_sum = sum(weights)
        preds = votes[votes > weight_sum * self.threshold]
        preds = preds.fillna(value=False)
        preds = preds.astype('bool')
        return preds[0].to_list()


def train_weighted_voting(weights, train_X, train_y, valid_X, valid_y, test_X, test_y,
                          threshold=0.5, mask=None):
    if mask:
        train_X = mask_columns(train_X, mask)
        valid_X = mask_columns(valid_X, mask)
        test_X = mask_columns(test_X, mask)
    model = WeightedVoting(weights, threshold)
    train_pred = model.predict(train_X)
    valid_pred = model.predict(valid_X)
    test_pred = model.predict(test_X)
    return (get_metrics(train_y, train_pred, verbose=True),
            get_metrics(valid_y, valid_pred, verbose=True),
            get_metrics(test_y, test_pred, verbose=True))


def train_weighted_voting_repeatedly(train_X, train_y, valid_X, valid_y,
                                       test_X, test_y, threshold=0.5,
                                       masks_list=None):
    metrics_bundles = []
    for mask in masks_list:
        metrics_bundles.append(train_weighted_voting('f1score',
                                                     train_X, train_y,
                                                      valid_X, valid_y,
                                                      test_X, test_y,
                                                      mask=mask))
    train_metrics, valid_metrics, test_metrics = zip(*metrics_bundles)
    valid_test_accs = zip(list(valid_metrics), list(test_metrics))
    valid_test_acc_curves(valid_test_accs)
    get_avg_metrics(train_metrics)
    get_avg_metrics(valid_metrics)
    get_avg_metrics(test_metrics)