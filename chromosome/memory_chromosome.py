from chromosome.chromosome import Chromosome


class MemoryChromosome(Chromosome):
    """
    Type of Chromosome that remembers his previous scores, able to
    calculate its fitness' moving average.
    """
    def __init__(self, total_genes, fitness_function=None):
        super().__init__(total_genes, fitness_function)
        self.age = 0

    def register_fitness(self, new_fitness):
        """
        It would probably be better not to keep all previous fitnesses,
        but just the moving average and number of items (since it is
        sufficient for calculating next avg. fit. for new fit. measur.)
        """
        try:
            unfolded_sum = self.fitness_value * self.age
            self.fitness_value = (unfolded_sum + new_fitness) / (self.age + 1)
        except TypeError:
            self.fitness_value = new_fitness
        self.age += 1

    def calculate_fitness(self, master, train, valid):
        fitness = self.fitness_function(master, train, valid)
        # master.register_fitness(fitness)
        # train.register_fitness(fitness)
        # valid.register_fitness(-fitness)
        return fitness

    @staticmethod
    def calculate_fitness_wrapper(master, train, valid):
        master.calculate_fitness(master, train, valid)
