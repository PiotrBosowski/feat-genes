from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


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


def mask_rows(data, mask):
    return data[list(map(bool, mask))]
