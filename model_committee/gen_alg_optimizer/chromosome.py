import random


class Chromosome:
    def __init__(self, gene_map=None, genes_count=None, random_init=True):
        if gene_map:
            self.genes = gene_map
        else:
            self.genes = random.choices([0, 1], k=genes_count) \
                if random_init else [0] * genes_count
        self.fitness = None

    def __str__(self):
        fitness = f"{self.fitness:.4f}" if self.fitness else "0.0"
        return f"fit: {fitness}, {self.genes}, len: [{self.seq_len()}]"

    def fitness_seq_len_regularized(self):
        return self.fitness - 0.0001 * self.seq_len()  # todo: uncomment

    def mutate(self, genes_to_mutate=1, chernobyl=False):
        if chernobyl:
            genes_to_mutate *= 3  # important, too aggressive is bad
        genes_to_toggle = random.sample(range(len(self.genes)),
                                        genes_to_mutate)
        for gt in genes_to_toggle:
            self.genes[gt] = int(not bool(self.genes[gt]))

    def seq_len(self):
        return sum(self.genes)

    @staticmethod
    def born(mother, father, crossover_strategy):
        gene_maps = crossover_strategy(mother.genes, father.genes)
        newborns = [Chromosome(gene_map=gene_map) for gene_map in gene_maps]
        return newborns

    @staticmethod
    def two_point_crossover(mom_genes, dad_genes):
        """mothers and fathers genes lengths should be the same"""
        slices = (random.randrange(len(mom_genes)),
                  random.randrange(len(mom_genes)))
        slc1 = min(slices)
        slc2 = max(slices)
        first_child = mom_genes[:slc1] + dad_genes[slc1:slc2] + mom_genes[slc2:]
        secnd_child = dad_genes[:slc1] + mom_genes[slc1:slc2] + dad_genes[slc2:]
        return first_child, secnd_child

    @staticmethod
    def crossover_fn(survived, population_count):
        """
        crossover - swapping part of the chromosome with other (% of children
        that come from crossover, ~0.5)
        """
        offspring = []
        while len(survived) + len(offspring) < population_count:
            mother = survived[random.randrange(len(survived))]
            father = survived[random.randrange(len(survived))]
            children = Chromosome.born(mother, father, Chromosome.two_point_crossover)
            offspring.extend(children)
        if len(survived) + len(offspring) > population_count:
            offspring.pop()
        return offspring
