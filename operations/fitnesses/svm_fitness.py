from sklearn import svm, metrics

from utils.data_preparation import mask_columns


class SVMacc:
    """
    fitness function - evaluates the quality of the solution (chromosome)
    and saves it in the chromosome.fitness variable
    """

    def __init__(self, kernel, train_X, train_y, test_X, test_y):
        self.kernel = kernel
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def __call__(self, chrom):
        train_X = mask_columns(self.train_X, chrom.genes)
        test_X = mask_columns(self.test_X, chrom.genes)
        classifier = svm.SVC(kernel=self.kernel,
                             class_weight=None)
        classifier.fit(train_X, self.train_y.values.ravel())
        pred_y = classifier.predict(test_X)
        chrom.fitness = metrics.accuracy_score(self.test_y, pred_y)
        return chrom.fitness
