import random

from chromosome.chromosome import Chromosome


class Selection:
    """
    The purpose of this class is to select best chromosomes that will
    reproduce and compose the next epoch's population.
    """

    def __init__(self, surviving_chance):
        self.surviving_chance = surviving_chance

    def __call__(self, population: [Chromosome]) -> [Chromosome]:
        chroms_to_survive = round(len(population) * self.surviving_chance)
        population.sort(lambda chrom: chrom.fitness_value, reverse=True)
        survivors = [population.pop(min(random.randrange(len(population)),
                                        random.randrange(len(population))))
                     for _ in range(chroms_to_survive)]
        return survivors

