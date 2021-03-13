import random


def two_point_crossover(mom_genes, dad_genes):
    """mothers and fathers genes lengths should be the same"""
    slices = (random.randrange(len(mom_genes)),
              random.randrange(len(mom_genes)))
    slc1 = min(slices)
    slc2 = max(slices)
    first_child = mom_genes[:slc1] + dad_genes[slc1:slc2] + mom_genes[slc2:]
    secnd_child = dad_genes[:slc1] + mom_genes[slc1:slc2] + dad_genes[slc2:]
    return first_child, secnd_child


def crossover_fn(survived, population_count):
    """
    crossover - swapping part of the chromosome with other (% of children
    that come from crossover, ~0.5)
    """
    offspring = []
    while len(survived) + len(offspring) < population_count:
        mother = survived[random.randrange(len(survived))]
        father = survived[random.randrange(len(survived))]
        children = Chromosome.born(mother, father,
                                   Chromosome.two_point_crossover)
        offspring.extend(children)
    if len(survived) + len(offspring) > population_count:
        offspring.pop()
    return offspring


    @staticmethod
    def born(mother, father, crossover_strategy):
        gene_maps = crossover_strategy(mother.genes, father.genes)
        newborns = [Chromosome(gene_map=gene_map) for gene_map in gene_maps]
        return newborns