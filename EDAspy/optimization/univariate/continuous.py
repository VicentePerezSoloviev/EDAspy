import pandas as pd
import numpy as np

"""
In this version of UMDA, instead of a vector of probabilities, a vector of univariate normal distributions is found.
When sampling, it is sampled from gaussian distribution
"""


class UMDAc:

    """
    Univariate marginal Estimation of Distribution algorithm continuous.
    New individuals are sampled from a vector of univariate normal distributions. It can be used for hyper-parameter
    optimization or to optimize a function.

    ...

    Attributes:
    --------------------

    generation: pandas DataFrame
        Last generation of the algorithm.

    best_mae_global: float
        Best cost found.

    best_ind_global: pandas DataFrame
        First row of the pandas DataFrame. Can be casted to dictionary.

    history: list
        List of the costs found during runtime.

    SIZE_GEN: int
        Parameter set by user. Number of the individuals in each generation.

    MAX_ITER: int
        Parameter set by user. Maximum number of iterations of the algorithm.

    DEAD_ITER: int
        Parameter set by user. Number of iterations after which, if no improvement reached, the algorithm finishes.

    vector: pandas DataFrame
        When initialized, parameters set by the user. When finished, statistics learned by the user.

    cost_function:
        Set by user. Cost function set to optimize.



    """

    SIZE_GEN = -1
    MAX_ITER = -1
    DEAD_ITER = -1
    alpha = -1
    vector = -1

    generation = -1

    best_mae_global = -1
    best_ind_global = -1

    cost_function = -1
    history = []

    def __init__(self, SIZE_GEN, MAX_ITER, DEAD_ITER, ALPHA, vector, aim, cost_function):
        """
        Constructor of the optimizer class

        :param SIZE_GEN: total size of the generations in the execution of the algorithm
        :type SIZE_GEN: int
        :param MAX_ITER: total number of iterations in case that optimum is not yet found. If reached, the optimum found is returned
        :type MAX_ITER: int
        :param DEAD_ITER: total number of iteration with no better solution found. If reached, the optimum found is returned
        :type DEAD_ITER: int
        :param ALPHA: percentage of the generation tu take, in order to sample from them. The best individuals selection
        :type ALPHA: float [0-1]
        :param vector: vector of normal distributions to sample from
        :type vector: pandas dataframe with columns ['mu', 'std'] and optional ['min', 'max']
        :param aim: Represents the optimization aim.
        :type aim: 'minimize' or 'maximize'.
        :param cost_function: a callable function implemented by the user, to optimize.
        :type cost_function: callable function which receives a dictionary as input and returns a numeric

        :raises Exception: cost function is not callable
        """

        self.SIZE_GEN = SIZE_GEN
        self.MAX_ITER = MAX_ITER
        self.alpha = ALPHA
        self.vector = vector

        self.variables = list(vector.columns)

        if aim == 'minimize':
            self.aim = 'min'
            self.best_mae_global = 999999999999
        elif aim == 'maximize':
            self.aim = 'max'
            self.best_mae_global = -999999999999
        else:
            raise Exception('ERROR when setting aim of optimizer. Only "minimize" or "maximize" is possible')

        # check if cost_function is real
        if callable(cost_function):
            self.cost_function = cost_function
        else:
            raise Exception('ERROR setting cost function. The cost function must be a callable function')

        # self.DEAD_ITER must be fewer than MAX_ITER
        if DEAD_ITER >= MAX_ITER:
            raise Exception('ERROR setting DEAD_ITER. The dead iterations must be fewer than the maximum iterations')
        else:
            self.DEAD_ITER = DEAD_ITER

    # new individual
    def __new_individual__(self):
        """
        Sample a new individual from the vector of probabilities.

        :return: a dictionary with the new individual; with names of the parameters as keys and the values.
        :rtype: dict
        """

        dic = {}
        for var in self.variables:
            mu = int(self.vector.loc['mu', var])
            std = int(self.vector.loc['std', var])

            # if exists min o max restriction
            if 'max' in list(self.vector.index):
                maximum = int(self.vector.loc['max', var])
            else:
                maximum = 999999999999
            if 'min' in list(self.vector.index):
                minimum = int(self.vector.loc['min', var])
            else:
                minimum = -999999999999

            sample = np.random.normal(mu, std, 1)
            while sample < minimum or sample > maximum:
                sample = np.random.normal(mu, std, 1)

            dic[var] = sample[0]
        return dic

    # build a generation of size SIZE_GEN from prob vector
    def new_generation(self):
        """
        Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """

        gen = pd.DataFrame(columns=self.variables)
        while len(gen) < self.SIZE_GEN:
            individual = self.__new_individual__()
            gen = gen.append(individual, True)

            # drop duplicate individuals
            gen = gen.drop_duplicates()
            gen = gen.reset_index()
            del gen['index']

        self.generation = gen

    # truncate the generation at alpha percent
    def truncation(self):
        """
        Selection of the best individuals of the actual generation. Updates the generation by selecting the best
        individuals
        """

        length = int(self.SIZE_GEN * self.alpha)

        # depending on whether min o maw is wanted
        if self.aim == 'max':
            self.generation = self.generation.nlargest(length, 'cost')
        elif self.aim == 'min':
            self.generation = self.generation.nsmallest(length, 'cost')

    # check the MAE of each individual
    def __check_individual__(self, individual):
        """
        Check the cost of the individual in the cost function

        :param individual: dictionary with the parameters to optimize as keys and the value as values of the keys
        :type individual: dict
        :return: a cost evaluated in the cost function to optimize
        :rtype: float
        """

        cost = self.cost_function(individual)
        return cost

    # check each individual of the generation
    def check_generation(self):
        """
        Check the cost of each individual in the cost function implemented by the user, and updates the
        generation DataFrame
        """

        for ind in range(len(self.generation)):
            cost = self.__check_individual__(self.generation.loc[ind])
            # print('ind: ', ind, ' cost ', cost)
            self.generation.loc[ind, 'cost'] = cost

    # update the probability vector
    def update_vector(self):
        """
        From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions.
        """

        for var in self.variables:
            array = list(self.generation[var].values)

            # calculate mu and std from data
            from scipy.stats import norm
            mu, std = norm.fit(array)

            # std should never be 0
            if std < 1:
                std = 1

            # update the vector probabilities
            self.vector.loc['mu', var] = mu
            self.vector.loc['std', var] = std

    # intern function to compare local cost with global one
    def __compare_costs__(self, local):
        """
        Check if the local best cost is better than the global one

        :param local: local best cost
        :type local: float
        :return: True if is better, False if not
        :rtype: bool
        """

        if self.aim == 'min':
            return local <= self.best_mae_global
        else:
            return local >= self.best_mae_global

    # run the class to find the optimum
    def run(self, output=True):
        """
        Run method to execute the EDA algorithm

        :param output: True if wanted to print each iteration
        :type output: bool
        :return: best cost, best individual, history of costs along execution
        :rtype: float, pandas dataframe, list
        """

        not_better = 0
        for i in range(self.MAX_ITER):
            self.new_generation()
            self.check_generation()
            self.truncation()
            self.update_vector()

            if self.aim == 'min':
                best_mae_local = self.generation['cost'].min()
            else:
                best_mae_local = self.generation['cost'].max()

            self.history.append(best_mae_local)
            best_ind_local = self.generation[self.generation['cost'] == best_mae_local]

            # update the best values ever
            # if best_mae_local <= self.best_mae_global:
            if self.__compare_costs__(best_mae_local):
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0
            else:
                not_better = not_better + 1
                if not_better == self.DEAD_ITER:
                    return self.best_mae_global, self.best_ind_global, self.history

            if output:
                print('IT ', i, 'best cost ', best_mae_local)

        return self.best_mae_global, self.best_ind_global, self.history
