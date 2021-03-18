import multiprocessing

from supervisor.passive_supervisor import PassiveSupervisor


def lambda_replacement(master, train, valid):
    # master, train, valid = tup
    master.calculate_fitness(master, train, valid)


class ActiveSupervisor(PassiveSupervisor):
    """
    Former 'Supervisor'.
    A class that leads the whole evolution process, orchestrating the
    mainline evolution as well as secondary ones (performed passively
    by ReverseSupervisors).

    Evaluating fitness must be adaptive to the number of features and
    the size of train/valid dataset.
    """

    current_best = 0.

    def __init__(self, genes_count, population_count, fitness, selection,
                 breeding, mutation, cataclysm, train_data_provider,
                 valid_data_provider, running_condition, chromosome_type):
        super().__init__(genes_count, population_count, selection, breeding,
                         mutation, cataclysm, chromosome_type, fitness)
        self.train_data_provider = train_data_provider
        self.valid_data_provider = valid_data_provider
        self.running_condition = running_condition
        self.chromosome_type = chromosome_type

    def evaluate_fitness(self):
        """
        train_data_provider will try to give the highest score training
        data, while the valid_data_provider - the hardest possible
        validation set to make the task more challenging (hence
        possibly improving generalization).
        """
        threads_count = 4
        train_population = self.train_data_provider.step()
        valid_population = self.valid_data_provider.step()
        pool = multiprocessing.Pool(threads_count)
        # list(map(lambda_replacement,
        #          list(zip(self.population, train_population,
        #                   valid_population))))
        pool.starmap(lambda_replacement,  # todo: fix this multithreading goddamn
                     list(zip(self.population, train_population,
                          valid_population)))
        liczba = 1

    def run(self):
        self.evaluate_fitness()
        # f.e. number of max epochs exceeded or no improvement since n epochs
        while self.running_condition(self):
            print(f"Epoch [{self.epoch + 1}]")
            self.population = self.step()
            self.evaluate_fitness()
            # validation part
            master = self.get_best()
            train = self.train_data_provider.get_best()
            valid = self.valid_data_provider.get_best()
            complete = self.valid_data_provider.get_complete()
            current = master.fitness_function(master, train, complete)
            print(f"FIT: [{current:.4f}], "
                  f"master:[{master.fitness_value:.4f}]"
                  f"<avg.len: {100*self.avg_len()/len(master):.2f}%>, "
                  f"train:[{train.fitness_value:.4f}]"
                  f"<avg.len: {100*self.train_data_provider.avg_len()/len(train):.2f}%>, "
                  f"valid:[{valid.fitness_value:.4f}]"
                  f"<avg.len: {100*self.valid_data_provider.avg_len()/len(valid):.2f}%>")
            if current > ActiveSupervisor.current_best:
                ActiveSupervisor.current_best = current
                with open('results.txt', 'w') as file:
                    file.write(str(current))


    # def save_solution_if_better(self, chrom):
    #     # if better fit or shorter sequence
    #     if chrom.fitness > self.current_best_fit or \
    #             (chrom.fitness == self.current_best_fit and
    #              chrom.seq_len() < self.best_fit_seq_len):
    #         self.current_best_fit = chrom.fitness
    #         self.best_fit_seq_len = chrom.seq_len()
    #         with open(self.output_path, 'a') as log_file:
    #             print(str(chrom.genes), file=log_file)

    # print(f"Epoch [{self.epoch + 1}]:", end=' ')
