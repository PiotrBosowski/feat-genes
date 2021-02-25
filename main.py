from utils.data_preparation import prepare_data, get_models_count
from generator.chromosome import Chromosome
from fitnesses.fitness import XGBoostAcc
from generator.generator import GENerator

BINARY_valid_1000 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/valid/combined_outputs-2021-02-08_01-30-23_SOURCE_COLUMN.csv'
BINARY_test_15108 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/test/combined_outputs-2021-02-08_01-21-46_SOURCE_COLUMN.csv'
# 15108 split between 3000 and 12108:
BINARY_test_3000 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/train/combined_outputs_SOURCE_COLUMN.csv'
BINARY_test_12108 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/test/combined_outputs_SOURCE_COLUMN.csv'


if __name__ == '__main__':
    output_path = f'/home/peter/Desktop/Inzynierka/committee_outputs/finalne/results_xgboost_2.txt'

    train_X, train_y = prepare_data(BINARY_valid_1000)
    valid_X, valid_y = prepare_data(BINARY_test_3000)
    test_X, test_y = prepare_data(BINARY_test_12108)

    fitness_xgboost = XGBoostAcc(train_X, train_y, valid_X, valid_y)

    generator = GENerator(genes_count=get_models_count(BINARY_test_3000),
                          population_count=80,
                          fitness_fn=fitness_xgboost,
                          selection_fn=GENerator.selection_fn,
                          crossover_ratio=0.50,
                          crossover_fn=Chromosome.crossover_fn,
                          mutation_ratio=0.05,
                          elitism_ratio=0.07,
                          genes_to_mutate=0.07,
                          chernobyl_every=80,
                          output_path=f'/home/peter/Desktop/Inzynierka/committee_outputs/finalne/results_xgboost_4.txt',
                          num_def_siblings=0)
    generator.run(max_epochs=1000)
