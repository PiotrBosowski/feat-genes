import random
from multiprocessing.pool import ThreadPool

from chromosome.chromosome import Chromosome


class GENerator:
    def __init__(self, output_path, genes_count, population_count, fitness_fn,
                 selection_fn, crossover_ratio, crossover_fn, mutation_ratio,
                 elitism_ratio, genes_to_mutate=None, chernobyl_every=100,
                 num_def_siblings=1):
        self.current_best_fit = 0.0  # defaults
        self.best_fit_seq_len = 10000
        self.output_path = output_path
        self.num_def_siblings = num_def_siblings
        self.genes_count = genes_count
        self.population_count = population_count
        self.fitness_fn = fitness_fn
        self.selection_fn = selection_fn
        self.crossover_ratio = crossover_ratio
        self.crossover_fn = crossover_fn
        self.mutation_ratio = mutation_ratio
        self.train_data_provider = None
        self.valid_data_provider = None
        self.elitism_ratio = elitism_ratio
        self.running_condition = None
        self.epoch = None
        self.chernobyl_every = chernobyl_every
        self.genes_to_mutate = round(genes_count * genes_to_mutate) \
            if genes_to_mutate else 1
        self.population = [Chromosome(genes_count=genes_count)
                           for _ in range(population_count)]
        # number of elites that will survive every epoch with no fight
        self.elites_surviving = round(population_count * elitism_ratio)
        # number of non-elites that will survive each epoch:
        self.plebs_surviving = round(population_count * (1 - crossover_ratio)
                                     - self.elites_surviving)

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

    # evaluating fitness must be adaptive to the number of
    # features and the size of train/valid dataset
    def evaluate_fitness(self):
        # calculating fitness in parallel:
        # those providers will run their own evolution of provided
        # data
        train_data = self.train_data_provider()
        # train data will try to give the highest score (optimistic)
        # training data, while the valid one - pessimistic, to make
        # the task harder (hence to improve generalization)
        valid_data = self.valid_data_provider()
        pool = ThreadPool(8)
        pool.map(self.fitness_fn, self.population)

    def init_population(self):
        """
        initializes the population with random chromosomes
        """
        pass

    def run(self):
        self.epoch = 0
        self.init_population()
        self.evaluate_fitness()
        # f.e. number of max epochs exceeded or no improvement since n epochs
        while self.running_condition(self):
            print(f"Epoch [{self.epoch + 1}]:", end=' ')



            # # sorting by fitness descending
            # self.population.sort(
            #     key=lambda chrom: chrom.fitness_seq_len_regularized(),
            #     reverse=True)
            # save better results:
            # for chrom in reversed(self.population):
            #     self.save_solution_if_better(chrom)
            # print(f"max_fit: [{self.population[0].fitness:.5f}]", end=' ')
            # print(f"with seq_len of {sum(self.population[0].genes)},", end=' ')
            # print(f"avg_fit: [{self.avg_fit():.5f}]", end=' ')
            # print(f"avg_seq_len: [{self.avg_seq_len():.1f}]")
            chernobyl_difference = abs(self.population[0].fitness - self.avg_fit()) < 0.00040
            # elitism causes top elitism_ratio chromosomes to survive
            survived = self.population[:self.elites_surviving]
            self.population = self.population[self.elites_surviving:]
            survived.extend(GENerator.elite_defective_sibling(survived[0],
                                                              self.num_def_siblings))
            # the rest is fighting for survival:
            survived.extend(self.selection_fn(self.population,
                                              self.plebs_surviving - self.num_def_siblings))  # because of the sibling
            survived.extend(self.crossover_fn(survived,
                                              self.population_count))
            if self.chernobyl_every:
                self.perform_mutation(survived, chernobyl=((epoch + 1) % self.chernobyl_every == 0))
            else:
                self.perform_mutation(survived, chernobyl=chernobyl_difference)
            self.population = survived

            self.evaluate_fitness()

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
            if random.uniform(0, 1) < self.mutation_ratio or chernobyl:
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
