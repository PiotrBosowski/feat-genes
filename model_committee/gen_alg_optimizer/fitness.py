from sklearn import svm, metrics
import xgboost

import model_committee.settings as settings
from model_committee.data_preparation import mask_columns
from model_committee.naive_approaches.cumulative_voting import CumulativeVoting
from model_committee.naive_approaches.majority_voting import MajorityVoting
from model_committee.naive_approaches.weighted_voting import WeightedVoting


class SVMacc:
    """
    fitness function - evaluates the quality of the solution (chromosome)
    and saves it in the chromosome.fitness variable
    """

    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        train_X = mask_columns(self.train_X, chrom.genes)
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = svm.SVC(kernel=settings.current_kernel,
                             class_weight=None)
        classifier.fit(train_X, self.train_y.values.ravel())
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness


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


class CumulativeVotingAcc:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = CumulativeVoting()
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness


class MajorityVotingAcc:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = MajorityVoting()
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness


class WeightedVotingAcc:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = WeightedVoting()
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness