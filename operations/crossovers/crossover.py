import copy
import random


class TwoPointCrossover:
    """
    Returns a new chromosome made of 2 parents.
    """
    def __call__(self, mother, father):
        child = copy.deepcopy(mother)
        cut_1, cut_2 = random.sample(range(len(mother)), 2)
        if cut_1 > cut_2:
            cut_1, cut_2 = cut_2, cut_1
        child[cut_1:cut_2] = father[cut_1:cut_2]
        return child

# @staticmethod
# def elite_defective_sibling(chrom, num_defective_siblings, defects=1):
#     """Takes the best guy in the population and returns its defective
#     sibling - same genes except one missing"""
#     siblings = []
#     for i in range(num_defective_siblings):
#         active_indices = [ind for ind, g in enumerate(chrom.genes) if g]
#         genes_to_shutdown = random.choices(active_indices, k=defects)
#         new_genes = [int(g and ind not in genes_to_shutdown)
#                      for ind, g in enumerate(chrom.genes)]
#         siblings.append(Chromosome(new_genes))
#     return siblings
