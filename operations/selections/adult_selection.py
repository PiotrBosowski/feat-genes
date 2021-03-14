import random

from chromosome.chromosome import Chromosome


class AdultSelection:
    """
    The purpose of this class is to select best chromosomes that will
    reproduce and compose the next epoch's population.
    """

    def __init__(self, surviving_chance, maturity_age):
        self.surviving_chance = surviving_chance
        self.maturity_age = maturity_age

    def __call__(self, population: [Chromosome]) -> [Chromosome]:
        children = [p for p in population if p.age < self.maturity_age]
        adults = [p for p in population if p.age >= self.maturity_age]
        adults_to_survive = round(len(adults) * self.surviving_chance)
        adults.sort(key=lambda chrom: chrom.fitness_value, reverse=True)
        survivors = [adults.pop(min(random.randrange(len(adults)),
                                    random.randrange(len(adults))))
                     for _ in range(adults_to_survive)]
        return children + survivors
