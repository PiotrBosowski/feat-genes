import os
import pandas as pd

models_dir = r'/home/peter/media/temp-share/temp/kplabs/potatoes/' \
             r'models_fixed_datasplit_timeframe-0'


def get_all_model_paths():
    all_models = [os.path.join(models_dir, 'SingleModelKernel', model)
                  for model in os.listdir(os.path.join(models_dir,
                                                       'SingleModelKernel'))]
    return all_models


def load_potato_data():
    """Returns 4 datasets (test, train, valid, subvalid)"""
    test = pd.DataFrame()
    train = pd.DataFrame()
    valid = pd.DataFrame()
    subval = pd.DataFrame()
    for ind, model in enumerate(get_all_model_paths()):
        test_col = pd.read_csv(os.path.join(model, 'report-1', 'outputs.csv'))
        test[model] = test_col['prediction']
        valid_col = pd.read_csv(os.path.join(model, 'report-2', 'outputs.csv'))
        valid[model] = valid_col['prediction']
        subval_col = pd.read_csv(os.path.join(model, 'report-3', 'outputs.csv'))
        subval[model] = subval_col['prediction']
        train_col = pd.read_csv(os.path.join(model, 'report-4', 'outputs.csv'))
        train[model] = train_col['prediction']
        if ind == 0:
            test_label = test_col['label']
            valid_label = valid_col['label']
            subval_label = subval_col['label']
            train_label = train_col['label']
    return test, test_label, valid, valid_label,\
           subval, subval_label, train, train_label
