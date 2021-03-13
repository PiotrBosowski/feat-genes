from multiprocessing.pool import ThreadPool

from chromosome.chromosome import Chromosome


class Supervisor:
    """
    A class that leads the whole evolution process, orchestrating the
    mainline evolution as well as secondary ones (performed passively
    by ReverseSupervisors).

    Evaluating fitness must be adaptive to the number of features and
    the size of train/valid dataset.
    """
    def __init__(self, genes_count, population_count,
                 fitness, selection, crossover, mutation, cataclysm,
                 train_data_provider, valid_data_provider,
                 running_condition):
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
        self.epoch = 0
        self.population = [Chromosome(self.genes_count)
                           for _ in range(self.population_count)]

    def evaluate_fitness(self):
        # calculating fitness in parallel:
        # those providers will run their own evolution of provided
        # data
        train_subset = self.train_data_provider.step()
        # train data will try to give the highest score (optimistic)
        # training data, while the valid one - pessimistic, to make
        # the task harder (hence to improve generalization)
        valid_subset = self.valid_data_provider.step()
        pool = ThreadPool(8)
        pool.map(self.fitness, self.population)

    def run(self):
        self.evaluate_fitness()
        # f.e. number of max epochs exceeded or no improvement since n epochs
        while self.running_condition(self):
            self.cataclysm(self)
            print(f"Epoch [{self.epoch + 1}]:", end=' ')
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluate_fitness()

    def avg_fit(self):
        return sum([chrom.fitness_value for chrom in self.population]) \
               / self.population_count

    def avg_seq_len(self):
        return sum([chrom.active() for chrom in self.population]) \
               / self.population_count

    # def save_solution_if_better(self, chrom):
    #     # if better fit or shorter sequence
    #     if chrom.fitness > self.current_best_fit or \
    #             (chrom.fitness == self.current_best_fit and
    #              chrom.seq_len() < self.best_fit_seq_len):
    #         self.current_best_fit = chrom.fitness
    #         self.best_fit_seq_len = chrom.seq_len()
    #         with open(self.output_path, 'a') as log_file:
    #             print(str(chrom.genes), file=log_file)
