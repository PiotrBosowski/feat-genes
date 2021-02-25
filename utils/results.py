import csv
import os
import shutil
import pandas as pd
from collections import namedtuple

from sklearn import metrics

Metrics = namedtuple('Metrics', "acc rec prec f1 mcc wrong_preds")


def get_metrics(real, pred, verbose=True):
    output = Metrics(
        acc=metrics.accuracy_score(real, pred),
        rec=metrics.recall_score(real, pred),
        prec=metrics.precision_score(real, pred),
        f1=metrics.f1_score(real, pred),
        mcc=metrics.matthews_corrcoef(real, pred),
        wrong_preds=[list(real['label'])[i] != pred[i] for i, _ in enumerate(pred)]
    )
    log = f"acc: {output.acc:.5f}, " \
          f"rec: {output.rec:.5f}, " \
          f"prec: {output.prec:.5f}, " \
          f"f1: {output.f1:.5f}, " \
          f"mcc: {output.mcc:.5f}"
    if verbose:
        print(log)
    return output


def get_avg_metrics(metrics, verbose=True):
    accs, recs, precs, f1s, mccs, _ = zip(*metrics)
    avg_acc = sum(accs) / len(accs)
    avg_rec = sum(recs) / len(recs)
    avg_prec = sum(precs) / len(precs)
    avg_f1 = sum(f1s) / len(f1s)
    avg_mcc = sum(mccs) / len(mccs)
    if verbose:
        print("avg_acc:", avg_acc)
        print("avg_rec:", avg_rec)
        print("avg_prec:", avg_prec)
        print("avg_f1:", avg_f1)
        print("avg_mccs:", avg_mcc)
    return Metrics(acc=avg_acc,
                   rec=avg_rec,
                   prec=avg_prec,
                   f1=avg_f1,
                   mcc=avg_mcc,
                   wrong_preds=None)


    # train_stats, valid_stats, test_stats = train_svm("rbf", train_X, train_y, valid_X, valid_y,
    #                     test_X, test_y, mask=current_mask)
    # test['min'] = test_X.min(axis=1)
    # test['max'] = test_X.max(axis=1)
    # test['avg'] = test_X.mean(axis=1)
    # test['var'] = test_X.var(axis=1)
    # incorrect_preds = test[pd.Series(test_stats.wrong_preds)]
    # OR
    # train_X, train_y, train = prepare_data(train_4500, get_data=True)
    # test_X, test_y, test = prepare_data(valid_9000, get_data=True)
    # history = train_svm(train_X, test_X, train_y, test_y, 'rbf', current_mask)
    # print(history.test_y)
    # print(history.test_pred)
    # test['min'] = test_X.min(axis=1)
    # test['max'] = test_X.max(axis=1)
    # test['avg'] = test_X.mean(axis=1)
    # test['var'] = test_X.var(axis=1)
    # incorrect_preds = test[test['label'] != history.test_pred]
    #
    # output_path = '/home/peter/Desktop/wrong_predictions'
    # csv_path = os.path.join(output_path, "wrong_preds.csv")
    # incorrect_preds.to_csv(csv_path, float_format='%.3f')
    #
    # pull_incorrect_preds(csv_path, output_path,
    #                      '/home/peter/Desktop/datasets_svm_committee/14500_valid-20000_test/9000_rest-20000_test/rest_9000/origins.csv',
    #                      '/home/peter/covid/datasets/2.5k-1.5k-rest/test')


def get_wrong_preds(dataset, metrics):
    wrong_preds = dataset[pd.Series(metrics.wrong_preds)]
    return wrong_preds


def wrong_preds_to_csv(wrong_preds, output_path):
    wrong_preds.to_csv(output_path, float_format='%.3f')


def pull_wrong_images(incorrect_preds, output_path, origins_path,
                      dataset_path):
    with open(incorrect_preds, 'r') as preds_file, \
            open(origins_path, 'r') as origins_file:
        preds_reader = csv.DictReader(preds_file)
        preds = list(preds_reader)
        origins_reader = csv.DictReader(origins_file)
        preds_filenames = list(preds_reader.fieldnames)
        preds_filenames.append('origin')
        origins = list(origins_reader)
        for ind, pred in enumerate(preds):
            origin = next(o for o in origins
                          if o['in_dataset_path'] == pred['img_path'])
            preds[ind]['origin'] = origin['origin_datasource']
        with open(os.path.join(output_path,
                               'incorrect_preds_origins.csv'), 'w') \
                as output_file:
            writer = csv.DictWriter(output_file, fieldnames=preds_filenames)
            writer.writeheader()
            writer.writerows(preds)
        wrong_images_path = os.path.join(output_path, 'wrong_images')
        os.makedirs(wrong_images_path, exist_ok=True)
        for pred in preds:
            if pred['label'] == '0':
                out_path = os.path.join(wrong_images_path, pred['origin'],
                                        pred['img_path'])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                shutil.copy(os.path.join(dataset_path, pred['img_path']), out_path)

if __name__ == "__main__":
    pull_wrong_images('/home/peter/covid/wrong_predictions_BINARY/wrong_preds.csv',
                      '/home/peter/covid/wrong_predictions_BINARY/najnowsze_false_positive',
                      '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/test/origins.csv',
                      '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/test')