import math
import os

import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam

from utils.plotting import valid_test_acc_curves, plot_history_loss
from utils.results import get_metrics, get_avg_metrics
from utils.filesystem_saver import next_name





def train_neural_net(train_X, train_y, valid_X, valid_y, test_X, test_y,
                     plotting=False, save_model=False):
    # settings:
    path = '~/Desktop/keras_models'  # WRONG XD

    def scheduler(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * math.exp(-0.2)

    learning_rate = 0.0006
    epochs = 22
    batch_size = 8500
    model = Sequential()
    model.add(Input(shape=(len(train_X.columns),)))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1500, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(train_X, train_y, epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[lr_scheduler],
                        verbose=0,
                        validation_data=(valid_X, valid_y))
    train_pred = model.predict_classes(train_X)
    train_pred = [1 in lst for lst in train_pred]
    valid_pred = model.predict_classes(valid_X)
    valid_pred = [1 in lst for lst in valid_pred]
    test_pred = model.predict_classes(test_X)
    test_pred = [1 in lst for lst in test_pred]
    if plotting:
        plot_history_loss(history, "Artificial Neural Network")
    if save_model:
        model.save(os.path.join(path, next_name(path, "model-{0}")))
    return (get_metrics(train_y, train_pred, verbose=True),
            get_metrics(valid_y, valid_pred, verbose=True),
            get_metrics(test_y, test_pred, verbose=True))


def train_neural_net_repeatedly(train_X, train_y, valid_X, valid_y, test_X,
                                test_y, iterations=100):
    metrics_bundles = []
    for i in range(iterations):
        metrics_bundles.append(train_neural_net(train_X, train_y, valid_X,
                                                valid_y, test_X, test_y))

    train_metrics, valid_metrics, test_metrics = zip(*metrics_bundles)
    valid_test_accs = zip(list(valid_metrics), list(test_metrics))
    valid_test_acc_curves(valid_test_accs, "Artificial Neural Network")

    get_avg_metrics(train_metrics)
    get_avg_metrics(valid_metrics)
    get_avg_metrics(test_metrics)
