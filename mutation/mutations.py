import random


def mutate(self, genes_to_mutate=1, chernobyl=False):
    if chernobyl:
        genes_to_mutate *= 3  # important, too aggressive is bad
    genes_to_toggle = random.sample(range(len(self.genes)),
                                    genes_to_mutate)
    for gt in genes_to_toggle:
        self.genes[gt] = int(not bool(self.genes[gt]))


    # chernobyl_difference = abs(
    #     self.population[0].fitness - self.avg_fit()) < 0.00040
    # # elitism causes top elitism_ratio chromosomes to survive
    # survived = self.population[:self.elites_surviving]
    # self.population = self.population[self.elites_surviving:]
    # survived.extend(GENerator.elite_defective_sibling(survived[0],
    #                                                   self.num_def_siblings))
    # # the rest is fighting for survival:
    # survived.extend(self.selection_fn(self.population,
    #                                   self.plebs_surviving
    #                                   - self.num_def_siblings))  #
    #                                   because of the sibling
    # survived.extend(self.crossover_fn(survived,
    #                                   self.population_count))
    # if self.chernobyl_every:
    #     self.perform_mutation(survived, chernobyl=(
    #                 (epoch + 1) % self.chernobyl_every == 0))
    # else:
    #     self.perform_mutation(survived, chernobyl=chernobyl_difference)
    # self.population = survived

    def perform_mutation(self, population, chernobyl=False):
        """
        mutation - randomly toggling genes (mutation rate ~0.05 means that 5%
        of chromosomes will have a random gene toggled)
        """
        if chernobyl:
            print("Not great, not terrible, performing Chernobyl...")
        pop_iter = iter(population)
        # next(pop_iter)
        for chrom in pop_iter:
            if random.uniform(0, 1) < self.mutation or chernobyl:
                chrom.mutate(self.genes_to_mutate, chernobyl)
