from model_committee.data_preparation import prepare_data, get_models_count
from model_committee.gen_alg_optimizer.chromosome import Chromosome
from model_committee.gen_alg_optimizer.fitness import SVMacc, XGBoostAcc, \
    CumulativeVotingAcc, MajorityVotingAcc, WeightedVotingAcc
from model_committee.gen_alg_optimizer.generator import GENerator
from model_committee.settings import BINARY_valid_1000, BINARY_test_3000, \
    BINARY_test_12108

if __name__ == '__main__':
    output_path = f'/home/peter/Desktop/Inzynierka/committee_outputs/finalne/results_xgboost_2.txt'

    train_X, train_y = prepare_data(BINARY_valid_1000)
    valid_X, valid_y = prepare_data(BINARY_test_3000)
    test_X, test_y = prepare_data(BINARY_test_12108)

    fitness_svm = SVMacc(train_X, train_y, valid_X, valid_y)  # 500
    fitness_xgboost = XGBoostAcc(train_X, train_y, valid_X, valid_y)  # 150
    fitness_cumul_voting = CumulativeVotingAcc(train_X, train_y, valid_X, valid_y)  # 2000
    fitness_major_voting = MajorityVotingAcc(train_X, train_y, valid_X, valid_y)
    fitness_weighted_voting = WeightedVotingAcc(train_X, train_y, valid_X, valid_y)

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
    generator.run(max_epochs=10000)