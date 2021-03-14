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
