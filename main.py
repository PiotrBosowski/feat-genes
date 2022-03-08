from chromosome.decaying_chromosome import DecayingChromosome
from operations.breedings.adult_breeding import AdultBreeding
from operations.cataclysms.cataclysm import Cataclysm
from operations.crossovers.crossover import TwoPointCrossover
from operations.fitnesses.svm_fitness_aging import SVMaccAging
from operations.mutations.mutation import Mutation
from operations.selections.adult_selection import AdultSelection
from projects.covid import prepare_data, get_models_count
from supervisor.passive_supervisor import PassiveSupervisor
from supervisor.active_supervisor import ActiveSupervisor


BINARY_valid_1000 = r'/home/peter/media/temp-share/repositories/archiv/machine-learning/covid-19/data/committee_datasets/combined_outputs-2021-02-08_01-30-23_SOURCE_COLUMN_1000.csv'
BINARY_test_15108 = r'/home/peter/media/temp-share/repositories/archiv/machine-learning/covid-19/data/committee_datasetscombined_outputs-2021-02-08_01-21-46_SOURCE_COLUMN_15108.csv'
# 15108 split between 3000 and 12108:
BINARY_test_3000 = r'/home/peter/media/temp-share/repositories/archiv/machine-learning/covid-19/data/committee_datasets/combined_outputs_SOURCE_COLUMN_3000.csv'
BINARY_test_12108 = r'/home/peter/media/temp-share/repositories/archiv/machine-learning/covid-19/data/committee_datasets/combined_outputs_SOURCE_COLUMN_12108.csv'


if __name__ == '__main__':
    train_X, train_y = prepare_data(BINARY_valid_1000)
    valid_X, valid_y = prepare_data(BINARY_test_3000)
    test_X, test_y = prepare_data(BINARY_test_12108)

    # fitness_xgboost = XGBoostAcc(train_X, train_y, valid_X, valid_y)

    population_count = 400  # 400
    train_data_provider = PassiveSupervisor(genes_count=len(train_X),
                                            population_count=population_count,
                                            selection=AdultSelection(0.6, 7),
                                            breeding=AdultBreeding(
                                                TwoPointCrossover(), 7),
                                            mutation=Mutation(
                                                chrom_mut_chance=0.1,
                                                gen_mut_chance=0.1),
                                            cataclysm=Cataclysm(),
                                            chromosome_type=DecayingChromosome)

    valid_data_provider = PassiveSupervisor(genes_count=len(valid_X),
                                            population_count=population_count,
                                            selection=AdultSelection(0.6, 7),
                                            breeding=AdultBreeding(
                                                TwoPointCrossover(), 7),
                                            mutation=Mutation(
                                                chrom_mut_chance=0.1,
                                                gen_mut_chance=0.1),
                                            cataclysm=Cataclysm(),
                                            chromosome_type=DecayingChromosome)

    generator = ActiveSupervisor(genes_count=get_models_count(train_X),
                                 population_count=population_count,
                                 fitness=SVMaccAging(train_X=train_X,
                                                     train_y=train_y,
                                                     valid_X=valid_X,
                                                     valid_y=valid_y),
                                 selection=AdultSelection(0.6, 7),
                                 breeding=AdultBreeding(TwoPointCrossover(),
                                                        7),
                                 mutation=Mutation(chrom_mut_chance=0.1,
                                                   gen_mut_chance=0.1),
                                 cataclysm=Cataclysm(),
                                 train_data_provider=train_data_provider,
                                 valid_data_provider=valid_data_provider,
                                 running_condition=lambda _: True,
                                 chromosome_type=DecayingChromosome)
    generator.run()
