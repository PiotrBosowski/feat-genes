import random


def mutate(self, genes_to_mutate=1, chernobyl=False):
    if chernobyl:
        genes_to_mutate *= 3  # important, too aggressive is bad
    genes_to_toggle = random.sample(range(len(self.genes)),
                                    genes_to_mutate)
    for gt in genes_to_toggle:
        self.genes[gt] = int(not bool(self.genes[gt]))
