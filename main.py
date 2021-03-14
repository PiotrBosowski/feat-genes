from fitnesses.xgboost_fitness import XGBoostAcc
from supervisor.reverse_supervisor import ReverseSupervisor
from utils.data_preparation import prepare_data, get_models_count
from supervisor.supervisor import Supervisor

BINARY_valid_1000 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/valid/combined_outputs-2021-02-08_01-30-23_SOURCE_COLUMN.csv'
BINARY_test_15108 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/test/combined_outputs-2021-02-08_01-21-46_SOURCE_COLUMN.csv'
# 15108 split between 3000 and 12108:
BINARY_test_3000 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/train/combined_outputs_SOURCE_COLUMN.csv'
BINARY_test_12108 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/test/combined_outputs_SOURCE_COLUMN.csv'


if __name__ == '__main__':
    train_X, train_y = prepare_data(BINARY_valid_1000)
    valid_X, valid_y = prepare_data(BINARY_test_3000)
    test_X, test_y = prepare_data(BINARY_test_12108)

    fitness_xgboost = XGBoostAcc(train_X, train_y, valid_X, valid_y)

    train_data_provider = ReverseSupervisor(genes_count=len(train_X),
                                            population_count=100,
                                            selection=None,
                                            crossover=None,
                                            mutation=None,
                                            cataclysm=None)

    valid_data_provider = ReverseSupervisor(genes_count=len(valid_X),
                                            population_count=100,
                                            selection=None,
                                            crossover=None,
                                            mutation=None,
                                            cataclysm=None)

    generator = Supervisor(genes_count=get_models_count(train_X),
                           population_count=100,
                           fitness=None,
                           selection=None,
                           crossover=None,
                           mutation=None,
                           cataclysm=None,
                           train_data_provider=train_data_provider,
                           valid_data_provider=valid_data_provider,
                           running_condition=lambda: True)

    generator.run()
