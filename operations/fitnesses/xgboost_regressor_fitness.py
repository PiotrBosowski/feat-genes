import xgboost
from sklearn import metrics

from chromosome.chromosome import Chromosome
from utils.data_preparation import mask_columns, mask_rows


class XGBoostRegressorR2:
    def __init__(self, train_X, train_y, valid_X, valid_y):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.model = None

    def __call__(self,
                 master_chrom: Chromosome,
                 train_chrom: Chromosome,
                 valid_chrom: Chromosome):
        train_X = mask_columns(self.train_X, master_chrom.genes)
        valid_X = mask_columns(self.valid_X, master_chrom.genes)
        train_X = mask_rows(train_X, train_chrom.genes)
        train_y = mask_rows(self.train_y, train_chrom.genes)
        valid_X = mask_rows(valid_X, valid_chrom.genes)
        valid_y = mask_rows(self.valid_y, valid_chrom.genes)
        self.model = xgboost.XGBRegressor(tree_method='gpu_hist',
                                          predictor='gpu_predictor',
                                          n_jobs=4)
        self.model.learning_rate = 0.01
        self.model.n_estimators = 350
        self.model.subsample = 0.3
        self.model.fit(train_X, train_y.values.ravel())
        # predict causes error during multiprocessing; inplace_predict
        # is believed to be solving that issue
        pred_y = self.model.predict(valid_X)
        # booster = model.get_booster()
        # valid_X = cp.array(valid_X.to_numpy())
        # pred_y = booster.inplace_predict(valid_X).get()
        fitness = metrics.r2_score(valid_y, pred_y)
        return fitness
