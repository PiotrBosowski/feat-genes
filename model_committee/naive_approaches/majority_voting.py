from model_committee.data_preparation import mask_columns
from model_committee.plotting import valid_test_acc_curves
from model_committee.results import get_metrics, get_avg_metrics


class MajorityVoting:
    def __init__(self, vote_threshold=0.5, count_threshold=0.5):
        self.vote_threshold = vote_threshold
        self.count_threshold = count_threshold

    def predict(self, data_X):
        votes = data_X[data_X > self.vote_threshold]
        count = votes.notnull().sum(axis=1).to_frame()
        preds = count[count > (len(data_X.columns) * self.count_threshold)]
        preds = preds.fillna(value=False)
        preds = preds.astype('bool')
        return preds[0].to_list()


def train_majority_voting(train_X, train_y, valid_X, valid_y, test_X, test_y,
                          vote_threshold=0.5, count_threshold=0.5, mask=None):
    if mask:
        train_X = mask_columns(train_X, mask)
        valid_X = mask_columns(valid_X, mask)
        test_X = mask_columns(test_X, mask)
    model = MajorityVoting(vote_threshold, count_threshold)
    train_pred = model.predict(train_X)
    valid_pred = model.predict(valid_X)
    test_pred = model.predict(test_X)
    return (get_metrics(train_y, train_pred, verbose=True),
            get_metrics(valid_y, valid_pred, verbose=True),
            get_metrics(test_y, test_pred, verbose=True))

def train_majority_voting_repeatedly(train_X, train_y, valid_X, valid_y,
                                       test_X, test_y, threshold=0.5,
                                       masks_list=None):
    metrics_bundles = []
    for mask in masks_list:
        metrics_bundles.append(train_majority_voting(train_X, train_y,
                                                      valid_X, valid_y,
                                                      test_X, test_y,
                                                      mask=mask))
    train_metrics, valid_metrics, test_metrics = zip(*metrics_bundles)
    valid_test_accs = zip(list(valid_metrics), list(test_metrics))
    valid_test_acc_curves(valid_test_accs)
    get_avg_metrics(train_metrics)
    get_avg_metrics(valid_metrics)
    get_avg_metrics(test_metrics)