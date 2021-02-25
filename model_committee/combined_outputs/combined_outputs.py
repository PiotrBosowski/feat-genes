import os
import datetime

import pandas as pd

import settings
from domain.training_model import Training
import numpy as np
import csv


def first_item_raw(raw_output_vector):
    return raw_output_vector[0]


def max_item(raw_output_vector):
    return raw_output_vector.index(max(raw_output_vector))


def first_item_softmaxed(raw_output_vector):
    x = np.array(raw_output_vector)
    sm = np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
    return sm[0]


def first_item_normalized(raw_output_vector):
    x = np.array(raw_output_vector)
    return x[0] / np.sum(x)


def combined_outputs(paths, output_strategy=first_item_softmaxed,
                     label_mapping=None, test_set=False):
    models = [Training().load_from_path(path) for path in paths]
    structured_outputs = {}
    header = ['img_path', 'label', 'origin_datasource']
    for model in models:
        header.append(model.url())
        raw_outputs = model.get_last_validation().report.raw_outputs if not test_set\
            else model.get_last_report().assessment.raw_outputs
        for output in raw_outputs:
            if output.image.in_dataset_path not in structured_outputs:
                structured_outputs[output.image.in_dataset_path] = \
                    {"img_path": output.image.in_dataset_path,
                     "label": label_mapping[output.image.label],
                     'origin_datasource': output.image.origin_datasource}
            structured_outputs[output.image.in_dataset_path].update(
                {model.url(): output_strategy(output.outputs)})
    list_outputs = [otps for _, otps in structured_outputs.items()]
    now = datetime.datetime.now()
    filename = f'combined_outputs-{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    with open(os.path.join(settings.models_dir, filename), 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(list_outputs)
    return filename


def overview_csv(models=None):
    if not models:
        models = Training.load_all()
    models = Training.models_flat(models)
    header = ['architecture', 'model', 'f1score', 'recall', 'precision', 'bin_acc', 'acc', 'lr_scheduler', 'epochs', 'learning_rate']
    now = datetime.datetime.now()
    filename = f'overview-{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    filepath = os.path.join(settings.models_dir, filename)
    with open(filepath, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for model in models:
            current = {
                'architecture': model.session,
                'model': model.model,
                'f1score':
                    f"{model.get_last_validation().report.confusion.f1scores()[0]:.3f}",
                'recall':
                    f"{model.get_last_validation().report.confusion.recalls()[0]:.3f}",
                'precision':
                    f"{model.get_last_validation().report.confusion.precisions()[0]:.3f}",
                'bin_acc':
                    f"{model.get_last_validation().report.confusion.binary_accuracy():.3f}",
                'acc':
                    f"{model.get_last_validation().report.confusion.accuracy():.3f}",
                'lr_scheduler': str(model.parameters['lr_scheduler']),
                'epochs': str(model.parameters['epochs']),
                'learning_rate': str(model.parameters['learning_rate'])
            }
            writer.writerow(current)
    return filename


def correlation_matrix(combined_output_path, label_mapping=None):
    outputs = pd.read_csv(combined_output_path)
    if label_mapping:
        label_mapping = eval_mapping(label_mapping)
        outputs['label'] = outputs['label'].map(label_mapping)
    correlation = outputs.corr()
    now = datetime.datetime.now()
    filename = f'correlations-{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    correlation.to_csv(os.path.join(settings.models_dir, filename), float_format='%.3f')
    return filename


def eval_mapping(label_map):
    return {settings.all_labels.index(label): output
            for label, output in label_map.items()}


def split_combined_outputs_in_half(csv_path, output_path):
    """Because combined_outputs file keep the same order of rows as origins.csv
    files, and because rows of origins.csv are grouped by label and then by
    origin, we can just move all even rows to train, and all odd to valid set"""
    with open(csv_path, 'r') as input,\
         open(os.path.join(output_path, 'train', 'train.csv'), 'w') as train,\
         open(os.path.join(output_path, 'test', 'test.csv'), 'w') as test:
        reader = csv.DictReader(input)
        writer_train = csv.DictWriter(train, reader.fieldnames)
        writer_test = csv.DictWriter(test, reader.fieldnames)
        writer_train.writeheader()
        writer_test.writeheader()
        for ind, row in enumerate(reader):
            if ind % 2 == 0:
                writer_train.writerow(row)
            else:
                writer_test.writerow(row)


def combined_outputs_from_origins(origins_path, combined_outputs_path, output_path):
    with open(origins_path, 'r') as origins_file,\
         open(combined_outputs_path, 'r') as combined_file,\
         open(output_path, 'w') as output_file:
        origins_reader = csv.DictReader(origins_file)
        origins = list(origins_reader)
        combined_reader = csv.DictReader(combined_file)
        combined = list(combined_reader)
        header = combined_reader.fieldnames
        images = [img['in_dataset_path'] for img in origins]
        overlay = [comb for comb in combined if comb['img_path'] in images]
        output_writer = csv.DictWriter(output_file, header)
        output_writer.writeheader()
        output_writer.writerows(overlay)


def concatenate_outputs_csv(first_csv, secnd_csv, output_path):
    with open(first_csv, 'r') as first,\
         open(secnd_csv, 'r') as secnd,\
         open(output_path, 'w') as output:
        first_reader = csv.DictReader(first)
        secnd_reader = csv.DictReader(secnd)
        header = first_reader.fieldnames
        output_writer = csv.DictWriter(output, header)
        output_writer.writeheader()
        output_writer.writerows(first_reader)
        output_writer.writerows(secnd_reader)


if __name__ == '__main__':
    combined_outputs_from_origins('/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/test/origins.csv',
                                  '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/test/combined_outputs-2021-02-08_01-21-46_SOURCE_COLUMN.csv',
                                  '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/test/combined_outputs_SOURCE_COLUMN.csv')

    # label_mapping = {'covid': 1,
    #                  'non-covid': 0}
    # models = Training.get_all_modelpaths()
    # outputs_test = combined_outputs(models, label_mapping=label_mapping,
    #                                 test_set=False)
