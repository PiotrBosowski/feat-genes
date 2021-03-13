import random


class Chromosome:
    """
    Class that represents a regular chromosome in a genetic algorithm.
    It is a single solution to the problem being optimized.
    """
    def __init__(self, genes, fitness_function=None, random_init=True):
        self.fitness_function = fitness_function
        self.fitness_value = None
        if isinstance(genes, int):
            self.genes = random.choices([0, 1], k=genes) \
                if random_init else [0] * genes
        else:
            try:
                self.genes = [int(c) for c in genes]
            except RuntimeError:
                raise

    def __str__(self):
        fitness = self.get_fitness()
        return f"fit: {f'{fitness:.4f}' if fitness else 'NaN'}, " \
               f"len: [{self.active()}/{self.total()}], " \
               f"{str(self.genes)}"

    def active(self):
        return sum(self.genes)

    def total(self):
        return len(self.genes)

    def get_fitness(self):
        return self.fitness_value

    def calculate_fitness(self):
        """
        Raises an exception if no fitness_function has been provided.
        """
        self.register_fitness(self.fitness_function(self.genes))

    def register_fitness(self, new_fitness):
        self.fitness_value = new_fitness