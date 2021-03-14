import random


class Mutation:
    def __init__(self, chrom_mut_chance, gen_mut_chance):
        self.chrom_mut_chance = chrom_mut_chance
        self.gen_mut_chance = gen_mut_chance

    def __call__(self, population):
        chroms_to_mutate = random.sample(
            population,
            round(self.chrom_mut_chance * len(population)))
        for chrom in chroms_to_mutate:
            genes_to_mutate = random.sample(
                range(len(chrom)),
                round(self.gen_mut_chance * len(chrom)))
            for gt in genes_to_mutate:
                chrom[gt] = int(not bool(chrom[gt]))
        return population
