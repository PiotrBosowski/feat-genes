from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler


def load_data(data_path):
    data = read_csv(data_path)
    data_X = data.drop(['label', 'img_path'], axis=1)
    # for backward-compatibility reasons:
    if 'origin_datasource' in data.columns:
        data_X = data_X.drop(['origin_datasource'], axis=1)
    data_y = data[['label']]
    return data, data_X, data_y


def prepare_data(data_path, mask=None, verbose=False, get_data=False):
    data, data_X, data_y = load_data(data_path)
    data_X = mask_columns(data_X, mask, verbose=verbose)
    if get_data:
        return data_X, data_y, data
    return data_X, data_y


def scale_data(train_X, test_X):
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = DataFrame(scaler.transform(train_X))
    test_X = DataFrame(scaler.transform(test_X))
    return train_X, test_X


def mask_columns(data, mask, verbose=False):
    if mask:
        indices = [i for i, m in enumerate(mask) if not m]
        data = data.drop(data.columns[indices], axis=1)
        if verbose:
            for col in data.columns:
                print(col)
    return data


def get_models_count(data):
    if isinstance(data, DataFrame):
        return len(data.columns)
    data_X, _, _ = load_data(data)
    return len(data_X.columns)
