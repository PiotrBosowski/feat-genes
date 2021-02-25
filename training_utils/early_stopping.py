"""
Set of classes which help with performing an early stopping.
Early stopping condition is checked after each validation.
"""


class ValidLossBelow:
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return f"valid_loss<{self.threshold}"

    def __call__(self, assessment):
        return assessment.loss < self.threshold


class AccAbove:
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return f"acc>{self.threshold}"

    def __call__(self, assessment):
        acc = assessment.confusion.accuracy()
        return acc > self.threshold


class BinAccAbove:
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return f"bin_acc>{self.threshold}"

    def __call__(self, assessment):
        acc = assessment.confusion.binary_accuracy()
        return acc > self.threshold


class F1_0_Above:
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return f"F1>{self.threshold}"

    def __call__(self, assessment):
        f1 = assessment.confusion.f1scores()[0]
        return f1 > self.threshold
