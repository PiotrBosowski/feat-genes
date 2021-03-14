from sklearn import metrics

from classifiers.cumulative_voting import CumulativeVoting
from utils.data_preparation import mask_columns


class CumulativeVotingAcc:
    def __init__(self, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = CumulativeVoting()
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness

