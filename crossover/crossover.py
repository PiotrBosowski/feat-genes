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


@staticmethod
def selection_fn(population, plebs_surviving):
    """
    Selection - way of picking solutions (tournament, casino wheel) it
    gets whole population and returns
    hint: population is sorted by fitness
    current implementation: tournament without replacement
    """
    winners = []
    for i in range(plebs_surviving):
        # WINNER INDEX = MIN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        winner_index = min(random.randrange(len(population)),
                           random.randrange(len(population)))
        winners.append(population.pop(winner_index))
    return winners


@staticmethod
def elite_defective_sibling(chrom, num_defective_siblings, defects=1):
    """Takes the best guy in the population and returns its defective
    sibling - same genes except one missing"""
    siblings = []
    for i in range(num_defective_siblings):
        active_indices = [ind for ind, g in enumerate(chrom.genes) if g]
        genes_to_shutdown = random.choices(active_indices, k=defects)
        new_genes = [int(g and ind not in genes_to_shutdown)
                     for ind, g in enumerate(chrom.genes)]
        siblings.append(Chromosome(new_genes))
    return siblings
