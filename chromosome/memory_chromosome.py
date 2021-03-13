from chromosome.chromosome import Chromosome


class MemoryChromosome(Chromosome):
    """
    Type of Chromosome that remembers his previous scores, able to
    calculate its fitness' moving average.
    """
    def __init__(self, genes, fitness_function=None, random_init=True):
        super().__init__(genes, fitness_function, random_init)
        self.fitness_value = []

    def get_fitness(self):
        """
        It would probably be better not to keep all previous fitnesses,
        but just the moving average and number of items (since it is
        sufficient for calculating next avg. fit. for new fit. measur.)
        """
        return sum(self.fitness_value) / len(self.fitness_value)

    def register_fitness(self, new_fitness):
        self.fitness_value.append(new_fitness)
