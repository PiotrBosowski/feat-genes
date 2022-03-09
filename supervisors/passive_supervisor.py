import copy
from statistics import stdev
from random import shuffle



class PassiveSupervisor:
    """
    Former 'ReverseSupervisor'.
    The idea of this class is to be a reverse-supervisors of an evolution
    process - it has to maintain its evolution with just callbacks from
    outside and minimal feedback about the results (fitness).

    Idea: maybe Supervisor and ReverseSupervisor should have a common
    inheritance hierarchy?
    """

    def __init__(self, genes_count, population_count, selection, breeding,
                 mutation, cataclysm, chromosome_type, fitness=None):
        self.genes_count = genes_count
        self.population_count = population_count
        self.selection = selection
        self.breeding = breeding
        self.mutation = mutation
        self.cataclysm = cataclysm
        self.epoch = 0
        self.population = [chromosome_type(self.genes_count, fitness)
                           for _ in range(self.population_count)]

    def step(self):
        """
        Mirror of a Supervisor.run() function. We assume that fitnesses
        of the population are already calculated when entering the func.

        :return: Returning the population - supervisors will calculate
        and register (in each chromosome) its fitnesses as a 'side
        effect' of supervisors's own evolution process.
        """
        self.epoch += 1
        if self.cataclysm.check(self.population, self.epoch):
            self.population = self.cataclysm(self.population)
        else:
            survivors = self.selection(self.population)
            new_population = self.breeding(survivors, self.population_count)
            self.population = self.mutation(new_population)
        shuffle(self.population)
        return self.population

    def avg_fit(self):
        return sum([chrom.fitness_value for chrom in self.population]) \
               / self.population_count

    def stdev_fit(self):
        return stdev([chrom.fitness_value for chrom in self.population])

    def avg_len(self):
        return sum([chrom.active() for chrom in self.population]) \
               / (self.population_count * self.genes_count)

    def stdev_len(self):
        return stdev([chrom.active() for chrom in self.population]) \
               / self.genes_count

    def get_best(self):
        return sorted(self.population,
                      key=lambda chrom: chrom.fitness_value)[-1]

    def get_complete(self):
        ones = copy.deepcopy(self.population[0])
        ones.genes = [1] * len(ones.genes)
        return ones
