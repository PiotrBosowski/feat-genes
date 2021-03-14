import random


class Breeding:
    def __init__(self, crossover):
        self.crossover = crossover

    def __call__(self, survivors, desired_size):
        offspring = []
        while len(survivors) + len(offspring) < desired_size:
            mother, father = (random.sample(survivors, 2))
            children = self.crossover(mother, father)
            offspring.extend(children)
        return survivors + offspring
