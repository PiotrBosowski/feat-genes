import copy
import random


class AdultBreeding:
    def __init__(self, crossover, maturity_age):
        self.crossover = crossover
        self.maturity_age = maturity_age

    def breeding(self, survivors, desired_size):
        offspring = []
        adults = [s for s in survivors if s.age > self.maturity_age]
        while len(survivors) + len(offspring) < desired_size and adults:
            try:
                mother, father = (random.sample(adults, 2))
            except ValueError:
                mother = adults[0]
                father = copy.deepcopy(mother)
                father.shuffle()
            children = self.crossover(mother, father)
            offspring.extend(children)
        return survivors + offspring
