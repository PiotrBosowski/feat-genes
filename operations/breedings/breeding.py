import random


class Breeding:
    def __init__(self, crossover):
        self.crossover = crossover

    def breeding(self, parents, desired_size):
        offspring = []
        while len(parents) + len(offspring) < desired_size:
            mother, father = (random.sample(parents, 2))
            children = self.crossover(mother, father)
            offspring.extend(children)
        return parents + offspring
