import random


class Chromosome:
    """
    Class that represents a regular chromosome in a genetic algorithm.
    It is a single solution to the problem being optimized.
    """
    def __init__(self, total_genes, fitness_function=None):
        self.fitness_function = fitness_function
        self.fitness_value = None
        active_genes = random.uniform(0, total_genes)
        self.genes = [1 if i < active_genes else 0 for i in range(total_genes)]
        random.shuffle(self.genes)

    def active(self):
        return sum(self.genes)

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __str__(self):
        fit = self.fitness_value
        return f"fit: {f'{fit:.4f}' if fit else 'NaN'}, " \
               f"len: [{self.active()}/{len(self)}], " \
               f"{str(self.genes)}"

    def shuffle(self):
        random.shuffle(self.genes)

    # def calculate_fitness(self, *args):
    #     """
    #     Raises an exception if no fitness_function has been provided.
    #     """
    #     self.register_fitness(self.fitness_function(self.genes))

    def register_fitness(self, new_fitness):
        self.fitness_value = new_fitness
