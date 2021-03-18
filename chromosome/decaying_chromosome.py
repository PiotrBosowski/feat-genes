from chromosome.chromosome import Chromosome


class DecayingChromosome(Chromosome):
    """
    Type of Chromosome that remembers his previous scores, able to
    calculate its fitness' moving average.
    """
    # number of fitness records to be considered
    history_length = 4

    def __init__(self, total_genes, fitness_function=None):
        super().__init__(total_genes, fitness_function)
        self.age = 0
        self.history = []

    def register_fitness(self, new_fitness):
        """
        It would probably be better not to keep all previous fitnesses,
        but just the moving average and number of items (since it is
        sufficient for calculating next avg. fit. for new fit. measur.)
        """
        self.history.insert(0, new_fitness)
        self.history = self.history[0:DecayingChromosome.history_length]
        length = len(self.history)
        weights = [(2 * (length - i)) / (length * (length + 1))
                   for i in range(length)]
        self.fitness_value = sum([h * weights[i]
                                  for i, h in enumerate(self.history)])
        self.age += 1

    def calculate_fitness(self, master, train, valid):
        fitness = self.fitness_function(master, train, valid)
        master.register_fitness(fitness)
        train.register_fitness(fitness)
        valid.register_fitness(-fitness)  # we want valid set to be tough

    @staticmethod
    def calculate_fitness_wrapper(master, train, valid):
        master.calculate_fitness(master, train, valid)