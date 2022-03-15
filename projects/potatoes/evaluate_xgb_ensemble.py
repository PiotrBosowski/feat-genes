import os
import warnings

from pandas.errors import PerformanceWarning
from sklearn import metrics

from projects.potatoes.potatoes import load_potato_data
from utils.data_preparation import mask_columns

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)

import xgboost


def evaluate_saved_potato_model(model_path):
    # len([entry for entry in test_dataset.timeframe_id_mask if entry == 0])
    # 140
    # len([entry for entry in test_dataset.timeframe_id_mask if entry == 1])
    # 148
    # len([entry for entry in test_dataset.timeframe_id_mask if entry == 2])
    # 208
    # len([entry for entry in test_dataset.timeframe_id_mask if entry == 3])
    # 116

    column_mask = [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

    test_X, test_y, _, _, _, _, _, _ = load_potato_data()
    test_X = mask_columns(test_X, column_mask)
    test_X_tf0 = test_X[0:140]
    test_y_tf0 = test_y[0:140]
    test_X_tf1 = test_X[140:140 + 148]
    test_y_tf1 = test_y[140:140 + 148]
    test_X_tf2 = test_X[140 + 148: 140 + 148 + 208]
    test_y_tf2 = test_y[140 + 148: 140 + 148 + 208]
    test_X_tf3 = test_X[140 + 148 + 208: 140 + 148 + 208 + 116]
    test_y_tf3 = test_y[140 + 148 + 208: 140 + 148 + 208 + 116]
    model = xgboost.XGBRegressor()
    model.load_model(model_path)
    print('timeframe-0', metrics.r2_score(test_y_tf0,
                                          model.predict(test_X_tf0)))
    print('timeframe-1', metrics.r2_score(test_y_tf1,
                                          model.predict(test_X_tf1)))
    print('timeframe-2', metrics.r2_score(test_y_tf2,
                                          model.predict(test_X_tf2)))
    print('timeframe-3', metrics.r2_score(test_y_tf3,
                                          model.predict(test_X_tf3)))
    print('aggregated:', metrics.r2_score(test_y, model.predict(test_X)))


if __name__ == '__main__':
    test_X, test_y, valid_X, valid_y, \
    subval_X, subval_y, train_X, train_y = load_potato_data()

    evaluate_saved_potato_model(os.path.join('..', 'outputs',
                                             'experiment_7', '89', 'model.xgb'))
