import os

from data_preparation import prepare_data
from model_committee.gen_alg_optimizer.optimization_results import \
    BINARY_committee_results_svm_v2, xgboost_BINARY_committees, \
    cumulative_voting_BINARY_committees, weighted_voting_BINARY_committees, \
    majority_voting_BINARY_committees
from model_committee.naive_approaches.cumulative_voting import \
    train_cumulative_voting
from model_committee.naive_approaches.majority_voting import \
    train_majority_voting
from model_committee.naive_approaches.weighted_voting import \
    train_weighted_voting
from model_committee.neural_net_approach.neural_net_approach import \
    train_neural_net
from model_committee.results import get_wrong_preds
from model_committee.svm_approach.svm_approach import train_svm
from model_committee.xgboost_approach.xgboost_approach import train_xgboost
from settings import BINARY_valid_1000, BINARY_test_3000, BINARY_test_12108
from training_utils.confusion import Confusion

def per_source_stats(dataset, wrong_preds):
    dataset['wrong_pred'] = dataset['img_path'].isin(wrong_preds.img_path.to_list())
    sources = dataset['origin_datasource'].unique()
    confusion_per_src = {}
    for src in sources:
        src_images = dataset[(dataset['origin_datasource'] == src)]
        confusion_per_src[src] = Confusion.from_2d_list(
            [
                [len(src_images[(src_images['label'] == 1) & (src_images['wrong_pred'] == False)]),
                 len(src_images[(src_images['label'] == 0) & (src_images['wrong_pred'] == True)])],
                [len(src_images[(src_images['label'] == 1) & (src_images['wrong_pred'] == True)]),
                 len(src_images[(src_images['label'] == 0) & (src_images['wrong_pred'] == False)])],
            ], ['covid', 'non-covid'])

    for src, cnf in confusion_per_src.items():
        print(src)
        print(cnf)
        print("")
    return confusion_per_src


if __name__ == '__main__':
    liczba = 8
    train_X, train_y = prepare_data(BINARY_valid_1000)
    valid_X, valid_y = prepare_data(BINARY_test_3000)
    test_X, test_y, test = prepare_data(BINARY_test_12108, get_data=True)

    train_score, valid_score, test_score = train_xgboost(
                                                     train_X, train_y,
                                                     valid_X, valid_y,
                                                     test_X, test_y,
                                                     mask=None
    )

    wrong_preds = get_wrong_preds(test, test_score)
    pss = per_source_stats(test, wrong_preds)

    # number_of_iterations = 100
    # history = []
    # for i in range(number_of_iterations):
    #     print(f">>>>>>>>>>>>> ITERATION {i}")
    #     train_score, valid_score, test_score = train_neural_net(train_X, train_y,
    #                                                             valid_X, valid_y,
    #                                                             test_X, test_y)
    #     wrong_preds = get_wrong_preds(test, test_score)
    #     history.append(per_source_stats(test, wrong_preds))
    #
    # total = dict.fromkeys(history[0].keys())
    # for source, conf in total.items():
    #     total[source] = Confusion.empty(labels=['covid', 'non-covid'])
    # for hist in history:
    #     for key, item in hist.items():
    #         total[key] += item
    # print(f"AVG OVER {number_of_iterations} ITERATIONS:")
    # for source, conf in total.items():
    #     total[source] /= number_of_iterations
    #     print(source)
    #     print(total[source])
    #     print("")