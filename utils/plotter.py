import math
import os

import settings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

from domain.training_model import Training
from training_utils.confusion import Confusion
from utils.stopwatch import Stopwatch


class Plotter:

    @staticmethod
    def confusion(confusion, path):
        """
        Plots the confusion matrix and saves to the path.
        :param confusion: n-classes confusion matrix
        :param path: output file
        """
        with sns.plotting_context("notebook", font_scale=1.3):
            fig = plt.figure()
            x_axis_labels = y_axis_labels = confusion.labels
            # sns.set(font_scale=1.4) sets it globally
            ax = sns.heatmap(confusion.matrix, annot=True, cmap='Blues',
                             cbar=False, fmt='g', xticklabels=x_axis_labels,
                             yticklabels=y_axis_labels)
            ax.xaxis.set_ticks_position('top')
            fig.tight_layout()
            fig.savefig(os.path.join(path, "confusion_matrix.png"))
            plt.close(fig)
            # sns.set(font_scale=1.0) doesnt work like this

    @staticmethod
    def plot_loss(history, path):
        """
        Plots the loss and saves to the path.
        :param epoch_len: number of steps needed to complete one epoch
        :param history: history of training losses
        :param path: output file
        """
        epoch_len = max([row.step for row in history])
        history = [(row.epoch, row.step, row.report.loss,
                    row.train_loss)
                   for row in history]
        history = np.array(history)
        history[:, 1] += history[:, 0] * epoch_len
        fig = plt.figure()
        fig.suptitle('Train and Validation Losses vs. Epochs')
        ax1 = fig.add_subplot(111)
        ax1.plot(history[:, 1], history[:, 2], label='valid_loss')
        ax1.plot(history[:, 1], history[:, 3], label='train_loss')
        # ax1.set_ylim(0, 5.0)
        _, x_max_lim = ax1.get_xlim()
        ax1_xticks = [t for t in range(0, math.ceil(x_max_lim), epoch_len)]
        ax1_xticklabels = [str(t/epoch_len) for t in ax1_xticks]
        ax1.set_xticks(ax1_xticks)
        ax1.set_xticklabels(ax1_xticklabels)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(path, "loss.png"))
        plt.close(fig)

    @staticmethod
    def covid_rec_vs_acc(history, path):
        epoch_len = max([row.step for row in history])
        rec_acc = [(row.epoch, row.step,
                    row.report.confusion.recalls()[0],
                    row.report.confusion.accuracy())
                   for row in history]
        rec_acc = np.array(rec_acc)
        rec_acc[:, 1] += rec_acc[:, 0] * epoch_len
        fig = plt.figure()
        fig.suptitle('Covid Recall and Overall Accuracy vs. Epochs')
        ax1 = fig.add_subplot(111)
        ax1.plot(rec_acc[:, 1], rec_acc[:, 2], label='covid_recall')
        ax1.plot(rec_acc[:, 1], rec_acc[:, 3], label='overall_acc')
        ax1.set_ylim(0.0, 1.0)
        _, x_max_lim = ax1.get_xlim()
        ax1_xticks = [t for t in range(0, math.ceil(x_max_lim), epoch_len)]
        ax1_xticklabels = [str(t/epoch_len) for t in ax1_xticks]
        ax1.set_xticks(ax1_xticks)
        ax1.set_xticklabels(ax1_xticklabels)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(path, 'covid_recall.png'))
        plt.close(fig)

    @staticmethod
    def wrong_preds(wrong_preds, path, max_images=100):
        """
        Display and saves the set of images that were incorrectly predicted.
        :param max_images: max number of images to be plotted
        :param wrong_preds: wrong predictions [{img_path, real, pred}]
        :param path: output file
        """
        colors = ['red', 'orange', 'green', 'black']
        images_number = min(len(wrong_preds), max_images)
        fig = plt.figure(figsize=(10, 1 + math.ceil((images_number - 1) / 10)))
        for i, wrong_pred in enumerate(wrong_preds):
            if i >= images_number:
                break
            with Image.open(wrong_pred.path).convert('RGB') as image:
                image.thumbnail((128, 128), Image.ANTIALIAS)
                plt.subplot(math.ceil(images_number / 10),
                            10, i + 1, xticks=[], yticks=[])
                plt.imshow(image)
                pred_index = wrong_pred.pred
                real_index = settings.all_labels.index(wrong_pred.img.label)
                plt.xlabel(f"{settings.all_labels[real_index]} [T]",
                           color=colors[min(real_index, len(colors)-1)])
                plt.ylabel(f"{settings.all_labels[pred_index]} [F]",
                           color=colors[min(pred_index, len(colors)-1)])
        fig.tight_layout()
        fig.savefig(os.path.join(path, 'wrong_preds.png'))
        plt.close(fig)

    @staticmethod
    def sources_confusions(confusions_by_sources, output_path):
        with sns.plotting_context("notebook", font_scale=1.1):
            images_number = len(confusions_by_sources)
            inch_size = 2
            fig = plt.figure(figsize=(inch_size * 5, inch_size * (1 + math.ceil((images_number - 1) / 5))))
            for i, (source_name, source_conf) in enumerate(confusions_by_sources.items()):
                plt.subplot(math.ceil(images_number / 5), 5, i + 1)
                ax = sns.heatmap(source_conf.matrix, cmap='Blues', cbar=False, annot=True,
                                 fmt='g', yticklabels=False, xticklabels=False)
                ax.set_title(source_name[:13])
                plt.xlabel(f"bin_acc: {source_conf.binary_accuracy():.4f}")
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)

    @staticmethod
    def plot_dataset_structure(dataset):
        for label in dataset.labels:
            descriptions = dataset.sources.keys()
            sizes = [source[label] for source in dataset.sources.values()]
            explode = [0.05] * len(sizes)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, autopct='%1.1f%%', startangle=90, pctdistance=1.2)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.show()

    def plot(self, paths):
        models = [Training().load_from_path(path) for path in paths]
        for model in models:
            with Stopwatch("Plotter")(f"Plotting path {model.model_path}"):
                Plotter.plot_loss(model.history, model.model_path)
                Plotter.covid_rec_vs_acc(model.history, model.model_path)
                for report in model.reports:
                    Plotter.plot_report(report)

    @staticmethod
    def plot_report(report):
        Plotter.confusion(report.assessment.confusion,
                          report.path)
        Plotter.save_dataset_stats(list(report.datasets.values())[0], report.assessment,
                                   report.path)
        Plotter.wrong_preds(report.assessment.wrong_preds, report.path)

    @staticmethod
    def save_dataset_stats(dataset, assessment, output_path):
        confusions_by_sources = {}
        if not assessment.wrong_preds:
            return
        for source, labels_count in dataset['sources'].items():
            wrong_preds = [wp for wp in assessment.wrong_preds
                           if wp.img.origin_datasource == source]
            preds = [w.pred for w in wrong_preds]
            reals = [settings.all_labels.index(w.img.label)
                     for w in wrong_preds]
            confusions_by_sources[source] = Confusion.from_wrong_preds(
                settings.all_labels, preds, reals, labels_count)
        Plotter.sources_confusions(confusions_by_sources,
                                   os.path.join(output_path,
                                                'dataset_stats.png'))

