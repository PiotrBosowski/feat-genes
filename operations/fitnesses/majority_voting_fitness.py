from sklearn import metrics

from classifiers.majority_voting import MajorityVoting
from utils.data_preparation import mask_columns


class MajorityVotingAcc:
    def __init__(self, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = MajorityVoting()
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness


