import xgboost
from sklearn import metrics

from utils.data_preparation import mask_columns


class XGBoostAcc:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        train_X = mask_columns(self.train_X, chrom.genes)
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = xgboost.XGBClassifier(objective="binary:logistic",
                                           tree_method='gpu_hist',
                                           predictor='gpu_predictor')
        classifier.learning_rate = 0.01
        classifier.n_estimators = 350
        classifier.subsample = 0.3
        classifier.fit(train_X, self.train_y.values.ravel())
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness

