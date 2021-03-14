from operations.mutations.mutations import Mutation


class Cataclysm:
    def __init__(self):
        pass

    def check(self, population, epochs):
        return not epochs % 10

    def __call__(self, population):
        mutator = Mutation(chrom_mut_chance=0.75, gen_mut_chance=0.3)
        mutator(population)
