import numpy as np
import pandas as pd
from warnings import warn


class UMDAd:

    """
    Univariate discrete Estimation of Distribution algorithm.
    New individuals are sampled from a vector of univariate probabilities. It is a binary optimizer, than can be
    used for example for feature selection.
    ...
    Attributes:
    --------------------
    generation: pandas DataFrame
        Last generation of the algorithm.
    best_MAE_global: float
        Best cost found.
    best_ind_global: pandas DataFrame
        First row of the pandas DataFrame. Can be casted to dictionary.
    history: list
        List of the costs found during runtime.
    SIZE_GEN: int
        Parameter set by user. Number of the individuals in each generation.
    MAX_IT: int
        Parameter set by user. Maximum number of iterations of the algorithm.
    DEAD_ITER: int
        Parameter set by user. Number of iterations after which, if no improvement reached, the algorithm finishes.
    vector: pandas DataFrame
        When initialized, parameters set by the user. When finished, statistics learned by the user.
    cost_function:
        Set by user. Cost function set to optimize.
    """

    MAX_IT = -1
    DEAD_ITER = -1
    SIZE_GEN = -1
    ALPHA = -1
    vector = []
    variables = []
    cost_function = -1

    history = []

    generation = -1
    best_MAE_global = -1
    best_ind_global = -1

    # init function
    def __init__(self, MAX_IT, DEAD_ITER, SIZE_GEN, ALPHA, vector, cost_function, aim):
        """
        Constructor of the optimizer class.
        :param SIZE_GEN: total size of the generations in the execution of the algorithm
        :type SIZE_GEN: int
        :param MAX_IT: total number of iterations in case that optimum is not yet found. If reached, the optimum found is returned
        :type MAX_IT: int
        :param DEAD_ITER: total number of iteration with no better solution found. If reached, the optimum found is returned
        :type DEAD_ITER: int
        :param ALPHA: percentage of the generation tu take, in order to sample from them. The best individuals selection
        :type ALPHA: float [0-1]
        :param vector: vector of normal distributions to sample from
        :type vector: pandas dataframe with columns ['mu', 'std'] and optional ['max', 'min']
        :param aim: 'minimize' or 'maximize'. Represents the optimization aim.
        :type aim: string ['minimize' or 'maximize']
        :param cost_function: cost function to minimize
        :type cost_function: callable function which receives a dictionary as input and returns a numeric value
        :raises Exception: cost function is not callable
        """

        warn('This version of UMDAd is deprecated and will be removed in future version. Please consider '
             'using from EDAspy.optimization import UMDAd instead as a newer and optimized version of the algorithm',
             DeprecationWarning, stacklevel=2)

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
        """Sample a new individual from the vector of probabilities.
        :return: a dictionary with the new individual; with names of the parameters as keys and the values.
        :rtype: dict
        """

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
        """Build a new generation sampled from the vector of probabilities and updates the generation pandas dataframe
        """
        gen = pd.DataFrame(columns=self.variables)

        while len(gen) < self.SIZE_GEN:
            individual = self.__new_individual__()
            gen = gen.append(individual, True)

        self.generation = gen

    # check the cost of each individual of the generation
    def __check_individual__(self, individual):
        """Check the cost of the individual in the cost function
        :param individual: dictionary with the parameters to optimize as keys and the value as values of the keys
        :type individual: dict
        :return: a cost evaluated in the cost function to optimize.
        :rtype: float
        """

        cost = self.cost_function(individual)
        return cost

    # check the cost of each individual of the generation
    def check_generation(self):
        """Check the cost of each individual in the cost function implemented by the user
        """

        for ind in range(len(self.generation)):
            cost = self.__check_individual__(self.generation.loc[ind])
            self.generation.loc[ind, 'cost'] = cost

    # selection of the best individuals to mutate the next gen
    def individuals_selection(self):
        """Selection of the best individuals of the actual generation and updates the generation by selecting the best
        individuals
        """

        length = int(len(self.generation)*self.ALPHA)
        if self.aim == 'min':
            self.generation = self.generation.nsmallest(length, 'cost')
        else:
            self.generation = self.generation.nlargest(length, 'cost')

        self.generation = self.generation.reset_index()
        del self.generation['index']

    # based on the best individuals of the selection, rebuild the prob vector
    def update_vector(self):
        """From the best individuals update the vector of probabilities in order to the next generation can sample from
         it and update the vector of probabilities
        """

        for ind in self.variables:
            total = self.generation[ind].sum()
            prob = total / len(self.generation)
            self.vector[ind] = prob

    # intern function to compare local cost with global one
    def __compare_costs__(self, local):
        """Check if the local best cost is better than the global one
        :param local: local best cost
        :type local: float
        :return: True if is better, False if not
        :rtype: bool
        """

        if self.aim == 'min':
            return local < self.best_MAE_global
        else:
            return local > self.best_MAE_global

    # run method
    def run(self, output=True):
        """Run method to execute the EDA algorithm
        :param output: True if wanted to print each iteration
        :type output: bool
        :return: best cost, best individual, history of costs along execution
        :rtype: float, pandas dataframe, list
        """

        warn('This version of UMDAd is deprecated and will be removed in future version. Please consider '
             'using EDAspy.optimization.UMDAd instead as a newer and optimized version of the algorithm',
             DeprecationWarning, stacklevel=2)

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