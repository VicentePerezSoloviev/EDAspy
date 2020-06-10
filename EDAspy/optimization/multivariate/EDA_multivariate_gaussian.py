import pandas as pd
import numpy as np


class EDA_multivariate_gaussian:

    """Multivariate Estimation of Distribution algorithm continuous.
    New individuals are sampled from a multivariate normal distribution. Evidences are not allowed

    :param SIZE_GEN: total size of the generations in the execution of the algorithm
    :type SIZE_GEN: int
    :param MAX_ITER: total number of iterations in case that optimum is not yet found. If reached, the optimum found is returned
    :type MAX_ITER: int
    :param DEAD_ITER: total number of iteration with no better solution found. If reached, the optimum found is returned
    :type DEAD_ITER: int
    :param ALPHA: percentage of the generation tu take, in order to sample from them. The best individuals selection
    :type ALPHA: float [0-1]
    :param aim: Represents the optimization aim.
    :type aim: 'minimize' or 'maximize'.
    :param cost_function: a callable function implemented by the user, to optimize.
    :type cost_function: callable function which receives a dictionary as input and returns a numeric
    :param mus: pandas dataframe with initial mus of the multivariate gaussian
    :type mus: pandas dataframe (one row)
    :param sigma: pandas dataframe with the sigmas of the variable (diagonal of covariance matrix)
    :type sigma: pandas dataframe (one row)

    :raises Exception: cost function is not callable

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

    def __init__(self, SIZE_GEN, MAX_ITER, DEAD_ITER, ALPHA, aim, cost_function, mus, sigma):
        """Constructor of the optimizer class
        """

        self.SIZE_GEN = SIZE_GEN
        self.MAX_ITER = MAX_ITER
        self.alpha = ALPHA

        self.variables = list(sigma.columns)

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

        # multivariate
        self.mus = mus

        sigma_data = pd.DataFrame(columns=mus.columns)
        sigma_data['vars'] = list(sigma_data.columns)
        sigma_data = sigma_data.set_index('vars')
        for var in list(sigma_data.columns):
            sigma_data.loc[var, var] = float(sigma[var])
        sigma_data = sigma_data.fillna(0)

        self.sigma = sigma_data

    # new individual
    def __new_individual__(self):
        """Sample a new individual from the vector of probabilities.
        :return: a dictionary with the new individual; with names of the parameters as keys and the values.
        :rtype: dict
        """
        mus = self.mus.loc[0].values.tolist()
        sigma = self.sigma.values.tolist()

        rand = list(np.random.multivariate_normal(mus, sigma, 1)[0])
        dic = {}
        for i in range(len(rand)):
            key = list(self.sigma.columns)[i]
            dic[key] = rand[i]

        return dic

    # build a generation of size SIZE_GEN from prob vector
    def new_generation(self):
        """Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
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
        """ Selection of the best individuals of the actual generation. Updates the generation by selecting the best individuals
        """

        length = int(self.SIZE_GEN * self.alpha)

        # depending on whether min o maw is wanted
        if self.aim == 'max':
            self.generation = self.generation.nlargest(length, 'cost')
        elif self.aim == 'min':
            self.generation = self.generation.nsmallest(length, 'cost')

    # check the MAE of each individual
    def __check_individual__(self, individual):
        """Check the cost of the individual in the cost function

        :param individual: dictionary with the parameters to optimize as keys and the value as values of the keys
        :type individual: dict
        :return: a cost evaluated in the cost function to optimize
        :rtype: float
        """

        cost = self.cost_function(individual)
        return cost

    # check each individual of the generation
    def check_generation(self):
        """Check the cost of each individual in the cost function implemented by the user
        """

        for ind in range(len(self.generation)):
            cost = self.__check_individual__(self.generation.loc[ind])
            # print('ind: ', ind, ' cost ', cost)
            self.generation.loc[ind, 'cost'] = cost

    # update the probability vector
    def update_vector(self):
        """From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions
        """

        # build covariance matrix from selection
        self.variables = list(self.sigma.columns)
        self.generation = self.generation.astype(float)
        covariance_matrix = self.generation[self.variables].cov()  # covariance matrix
        self.sigma = covariance_matrix.copy()

        for var in self.variables:
            # change mean
            self.mus.loc[0, var] = float(self.generation[var].mean())

            # check if sigma has decreased in off
            if self.sigma.loc[var, var] <= 1:
                self.sigma.loc[var, var] = 1

    # intern function to compare local cost with global one
    def __compare_costs__(self, local):
        """Check if the local best cost is better than the global one
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
        """Run method to execute the EDA algorithm

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
