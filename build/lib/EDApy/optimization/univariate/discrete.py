import numpy as np
import pandas as pd


class UMDAd:
    MAX_IT = -1
    DEAD_ITER = -1
    SIZE_GEN = -1
    ALPHA = -1
    vector = []
    variables = []
    cost_function = -1
    esc = 15

    history = []

    generation = -1
    best_MAE_global = -1
    best_ind_global = -1

    # init function
    def __init__(self, MAX_IT, DEAD_ITER, SIZE_GEN, ALPHA, vector, cost_function, aim):
        self.ALPHA = ALPHA
        self.SIZE_GEN = SIZE_GEN
        self.MAX_IT = MAX_IT

        self.vector = vector
        self.variables = list(vector.columns)

        # check if cost_function is real
        if callable(cost_function):
            self.cost_function = cost_function
        else:
            raise Exception('ERROR setting cost function. The cost function must be a callable function')

        if aim == 'minimize':
            self.aim = 'min'
            self.best_MAE_global = 9999999999
        elif aim == 'maximize':
            self.aim = 'max'
            self.best_MAE_global = -9999999999
        else:
            raise Exception('ERROR when setting aim of optimizer. Only "minimize" or "maximize" is possible')

        # self.DEAD_ITER must be fewer than MAX_ITER
        if DEAD_ITER >= MAX_IT:
            raise Exception(
                'ERROR setting DEAD_ITER. The dead iterations must be fewer than the maximum iterations')
        else:
            self.DEAD_ITER = DEAD_ITER

    # new individual
    def __new_individual__(self):
        num_vars = len(self.variables)
        sample = list(np.random.uniform(low=0, high=1, size=num_vars))
        individual = {}
        index = 0
        for ind in self.variables:
            if float(self.vector[ind]) >= sample[index]:
                individual[ind] = 1
            else:
                individual[ind] = 0
            index = index + 1
        return individual

    # new generation
    def new_generation(self):
        gen = pd.DataFrame(columns=self.variables)

        while len(gen) < self.SIZE_GEN:
            individual = self.__new_individual__()
            gen = gen.append(individual, True)

        self.generation = gen

    # check the cost of each individual of the generation
    def __check_individual__(self, individual):
        cost = self.cost_function(individual)
        return cost

    # check the cost of each individual of the generation
    def check_generation(self):
        for ind in range(len(self.generation)):
            cost = self.__check_individual__(self.generation.loc[ind])
            self.generation.loc[ind, 'cost'] = cost

    # selection of the best individuals to mutate the next gen
    def individuals_selection(self):
        length = int(len(self.generation)*self.ALPHA)
        if self.aim == 'min':
            self.generation = self.generation.nsmallest(length, 'cost')
        else:
            self.generation = self.generation.nlargest(length, 'cost')

        self.generation = self.generation.reset_index()
        del self.generation['index']

    # based on the best individuals of the selection, rebuild the prob vector
    def update_vector(self):
        for ind in self.variables:
            total = self.generation[ind].sum()
            prob = total / len(self.generation)
            self.vector[ind] = prob

    # intern function to compare local cost with global one
    def __compare_costs__(self, local):
        """
        Check if the local best cost is better than the global one
        :param local: local best cost
        :return: True if is better, False if not
        """

        if self.aim == 'min':
            return local < self.best_MAE_global
        else:
            return local > self.best_MAE_global

    # run method
    def run(self, output=True):
        dead_iter = 0
        for i in range(self.MAX_IT):
            self.new_generation()
            self.check_generation()
            self.individuals_selection()
            self.update_vector()

            if self.aim == 'min':
                best_mae_local = self.generation['cost'].min()
            else:
                best_mae_local = self.generation['cost'].max()

            self.history.append(best_mae_local)
            best_ind_local = []
            best = self.generation[self.generation['cost'] == best_mae_local].loc[0]

            for j in self.variables:
                if int(best[j]) == 1:
                    best_ind_local.append(j)

            # update best of model
            if self.__compare_costs__(best_mae_local):
                self.best_MAE_global = best_mae_local
                self.best_ind_global = best_ind_local
                dead_iter = 0
            else:
                dead_iter = dead_iter + 1
                if dead_iter == self.DEAD_ITER:
                    return self.best_MAE_global, self.best_ind_global, self.history

            if output:
                print('IT ', i, 'best cost ', best_mae_local)
                print(best_ind_local)

        return self.best_MAE_global, self.best_ind_global, self.history
