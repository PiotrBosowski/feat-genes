import random
from multiprocessing.pool import ThreadPool

from chromosome.chromosome import Chromosome


class Supervisor:
    """
    A class that leads the whole evolution process, orchestrating the
    mainline evolution as well as secondary ones (performed passively
    by ReverseSupervisors).
    """
    def __init__(self, genes_count, population_count,
                 fitness, selection, crossover, mutation, cataclysm,
                 train_data_provider, valid_data_provider,
                 running_condition, cataclysm_condition):
        self.genes_count = genes_count
        self.population_count = population_count
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.cataclysm = cataclysm
        self.train_data_provider = train_data_provider
        self.valid_data_provider = valid_data_provider
        self.running_condition = running_condition
        self.cataclysm_condition = cataclysm_condition
        self.epoch = None
        self.population = []

    # evaluating fitness must be adaptive to the number of
    # features and the size of train/valid dataset
    def evaluate_fitness(self):
        # calculating fitness in parallel:
        # those providers will run their own evolution of provided
        # data
        train_subset = self.train_data_provider.get_data()
        # train data will try to give the highest score (optimistic)
        # training data, while the valid one - pessimistic, to make
        # the task harder (hence to improve generalization)
        valid_subset = self.valid_data_provider.get_data()
        pool = ThreadPool(8)
        pool.map(self.fitness, self.population)

    def init_population(self):
        """
        initializes the population with random chromosomes
        """
        self.population = [Chromosome(self.genes_count)
                           for _ in range(self.population_count)]

    def run(self):
        self.epoch = 0
        self.init_population()
        self.evaluate_fitness()
        # f.e. number of max epochs exceeded or no improvement since n epochs
        while self.running_condition(self):
            if self.cataclysm_condition(self):
                self.cataclysm()
            print(f"Epoch [{self.epoch + 1}]:", end=' ')
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluate_fitness()

    # chernobyl_difference = abs(
    #     self.population[0].fitness - self.avg_fit()) < 0.00040
    # # elitism causes top elitism_ratio chromosomes to survive
    # survived = self.population[:self.elites_surviving]
    # self.population = self.population[self.elites_surviving:]
    # survived.extend(GENerator.elite_defective_sibling(survived[0],
    #                                                   self.num_def_siblings))
    # # the rest is fighting for survival:
    # survived.extend(self.selection_fn(self.population,
    #                                   self.plebs_surviving
    #                                   - self.num_def_siblings))  #
    #                                   because of the sibling
    # survived.extend(self.crossover_fn(survived,
    #                                   self.population_count))
    # if self.chernobyl_every:
    #     self.perform_mutation(survived, chernobyl=(
    #                 (epoch + 1) % self.chernobyl_every == 0))
    # else:
    #     self.perform_mutation(survived, chernobyl=chernobyl_difference)
    # self.population = survived

    def perform_mutation(self, population, chernobyl=False):
        """
        mutation - randomly toggling genes (mutation rate ~0.05 means that 5%
        of chromosomes will have a random gene toggled)
        """
        if chernobyl:
            print("Not great, not terrible, performing Chernobyl...")
        pop_iter = iter(population)
        # next(pop_iter)
        for chrom in pop_iter:
            if random.uniform(0, 1) < self.mutation or chernobyl:
                chrom.mutate(self.genes_to_mutate, chernobyl)

    def save_solution_if_better(self, chrom):
        # if better fit or shorter sequence
        if chrom.fitness > self.current_best_fit or \
                (chrom.fitness == self.current_best_fit and
                 chrom.seq_len() < self.best_fit_seq_len):
            self.current_best_fit = chrom.fitness
            self.best_fit_seq_len = chrom.seq_len()
            with open(self.output_path, 'a') as log_file:
                print(str(chrom.genes), file=log_file)

    @staticmethod
    def selection_fn(population, plebs_surviving):
        """
        Selection - way of picking solutions (tournament, casino wheel) it
        gets whole population and returns
        hint: population is sorted by fitness
        current implementation: tournament without replacement
        """
        winners = []
        for i in range(plebs_surviving):
            # WINNER INDEX = MIN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            winner_index = min(random.randrange(len(population)),
                               random.randrange(len(population)))
            winners.append(population.pop(winner_index))
        return winners

    def avg_fit(self):
        return sum([chrom.fitness for chrom in self.population]) \
               / self.population_count

    def avg_seq_len(self):
        return sum([chrom.seq_len() for chrom in self.population]) \
               / self.population_count

    @staticmethod
    def elite_defective_sibling(chrom, num_defective_siblings, defects=1):
        """Takes the best guy in the population and returns its defective
        sibling - same genes except one missing"""
        siblings = []
        for i in range(num_defective_siblings):
            active_indices = [ind for ind, g in enumerate(chrom.genes) if g]
            genes_to_shutdown = random.choices(active_indices, k=defects)
            new_genes = [int(g and ind not in genes_to_shutdown)
                         for ind, g in enumerate(chrom.genes)]
            siblings.append(Chromosome(new_genes))
        return siblings
