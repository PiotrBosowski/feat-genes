import matplotlib.pyplot as plt


def valid_test_acc_curves(valid_test_metrics, title):
    valid_test_accs = [(m[0].acc, m[1].acc) for m in valid_test_metrics]
    valid, test = zip(*sorted(valid_test_accs, key=lambda k: k[0]))
    plt.plot(valid)
    plt.plot(test)
    plt.title(f'{title}: accuracies on valid and test sets')
    plt.ylabel('accuracy')
    plt.legend(['valid', 'test'], loc='upper left')
    plt.show()


def plot_history_loss(history, title):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='center right')
    plt.show()
