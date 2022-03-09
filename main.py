import warnings

from pandas.errors import PerformanceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)

from chromosome.decaying_chromosome import DecayingChromosome
from operations.breedings.adult_breeding import AdultBreeding
from operations.cataclysms.cataclysm import Cataclysm
from operations.crossovers.crossover import TwoPointCrossover
from operations.fitnesses.xgboost_regressor_fitness import XGBoostRegressorR2
from operations.mutations.mutation import Mutation
from operations.selections.adult_selection import AdultSelection
from projects.covid import get_models_count
from projects.potatoes import load_potato_data
from supervisors.passive_supervisor import PassiveSupervisor
from supervisors.active_supervisor import ActiveSupervisor


if __name__ == '__main__':
    test_X, test_y, train_X, train_y, \
    valid_X, valid_y, subval_X, subval_y = load_potato_data()

    # fitness_xgboost = XGBoostAcc(train_X, train_y, valid_X, valid_y)

    population_count = 50  # 400
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
                                 fitness=XGBoostRegressorR2(train_X=train_X,
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
