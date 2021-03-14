from random import shuffle

from chromosome.memory_chromosome import MemoryChromosome


class ReverseSupervisor:
    """
    The idea of this class is to be a reverse-supervisor of an evolution
    process - it has to maintain its evolution with just callbacks from
    outside and minimal feedback about the results (fitness).

    Idea: maybe Supervisor and ReverseSupervisor should have a common
    inheritance hierarchy?
    """

    def __init__(self, genes_count, population_count,
                 selection, crossover, mutation, cataclysm):
        self.genes_count = genes_count
        self.population_count = population_count
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.cataclysm = cataclysm
        self.epoch = 0
        # self.population = [MemoryChromosome(self.genes_count)] \
        #                   * self.population_count
        self.population = [MemoryChromosome(self.genes_count)
                           for _ in range(self.population_count)]

    def step(self):
        """
        Mirror of a Supervisor.run() function. We assume that fitnesses
        of the population are already calculated when entering the func.

        :return: Returning the population - supervisor will calculate
        and register (in each chromosome) its fitnesses as a 'side
        effect' of supervisor's own evolution process.
        """
        self.epoch += 1
        self.cataclysm()
        self.crossover()
        self.mutation()
        return shuffle(self.population)
