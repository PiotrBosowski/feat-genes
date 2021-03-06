from chromosome.memory_chromosome import MemoryChromosome
from operations.breedings.adult_breeding import AdultBreeding
from operations.cataclysms.cataclysm import Cataclysm
from operations.crossovers.crossover import TwoPointCrossover
from operations.fitnesses.svm_fitness_aging import SVMaccAging
from operations.mutations.mutation import Mutation
from operations.selections.adult_selection import AdultSelection
from supervisor.passive_supervisor import PassiveSupervisor
from utils.data_preparation import prepare_data, get_models_count
from supervisor.active_supervisor import ActiveSupervisor

BINARY_valid_1000 = r'C:\Users\piotr\Desktop\committee_datasets\combined_outputs-2021-02-08_01-30-23_SOURCE_COLUMN_1000.csv'
BINARY_test_15108 = r'C:\Users\piotr\Desktop\committee_datasets\combined_outputs-2021-02-08_01-21-46_SOURCE_COLUMN_15108.csv'
# 15108 split between 3000 and 12108:
BINARY_test_3000 = r'C:\Users\piotr\Desktop\committee_datasets\combined_outputs_SOURCE_COLUMN_3000.csv'
BINARY_test_12108 = r'C:\Users\piotr\Desktop\committee_datasets\combined_outputs_SOURCE_COLUMN_12108.csv'

if __name__ == '__main__':
    train_X, train_y = prepare_data(BINARY_valid_1000)
    valid_X, valid_y = prepare_data(BINARY_test_3000)
    test_X, test_y = prepare_data(BINARY_test_12108)

    # fitness_xgboost = XGBoostAcc(train_X, train_y, valid_X, valid_y)

    train_data_provider = PassiveSupervisor(genes_count=len(train_X),
                                            population_count=100,
                                            selection=AdultSelection(0.6, 4),
                                            breeding=AdultBreeding(
                                                TwoPointCrossover(), 4),
                                            mutation=Mutation(
                                                chrom_mut_chance=0.1,
                                                gen_mut_chance=0.1),
                                            cataclysm=Cataclysm(),
                                            chromosome_type=MemoryChromosome)

    valid_data_provider = PassiveSupervisor(genes_count=len(valid_X),
                                            population_count=100,
                                            selection=AdultSelection(0.6, 4),
                                            breeding=AdultBreeding(
                                                TwoPointCrossover(), 4),
                                            mutation=Mutation(
                                                chrom_mut_chance=0.1,
                                                gen_mut_chance=0.1),
                                            cataclysm=Cataclysm(),
                                            chromosome_type=MemoryChromosome)

    generator = ActiveSupervisor(genes_count=get_models_count(train_X),
                                 population_count=100,
                                 fitness=SVMaccAging(train_X=train_X,
                                                     train_y=train_y,
                                                     valid_X=valid_X,
                                                     valid_y=valid_y),
                                 selection=AdultSelection(0.6, 4),
                                 breeding=AdultBreeding(TwoPointCrossover(),
                                                        4),
                                 mutation=Mutation(chrom_mut_chance=0.1,
                                                   gen_mut_chance=0.1),
                                 cataclysm=Cataclysm(),
                                 train_data_provider=train_data_provider,
                                 valid_data_provider=valid_data_provider,
                                 running_condition=lambda _: True,
                                 chromosome_type=MemoryChromosome)
    generator.run()
