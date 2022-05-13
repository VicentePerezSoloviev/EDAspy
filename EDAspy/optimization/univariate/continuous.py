import pandas as pd
import numpy as np
from scipy.stats import norm


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

    size_gen: int
        Parameter set by user. Number of the individuals in each generation.

    max_iter: int
        Parameter set by user. Maximum number of iterations of the algorithm.

    dead_iter: int
        Parameter set by user. Number of iterations after which, if no improvement reached, the algorithm finishes.

    vector: pandas DataFrame
        When initialized, parameters set by the user. When finished, statistics learned by the user.

    cost_function:
        Set by user. Cost function set to optimize.
    """
    best_mae_global = -1
    best_ind_global = -1

    history = []

    def __init__(self, size_gen, max_iter, dead_iter, alpha, vector, std_bound=0.3, elite_factor=0.4):
        """
        Constructor of the UMDAc optimizer class

        Parameters
        ----------
            size_gen : int
                Population size of the algorithm (number of individuals per generation)
            max_iter : int
                Maximum number of iterations
            dead_iter : int
                Number of iterations after which, if no improvement is found, the runtime finish
            alpha : float
                Percentage ([0, 1]) of the population considered for the elite selection
            vector : DataFrame
                Pandas DataFrame with mandatory rows 'mu' and 'std' and names of variables as columns. The values
                in initialization are used to sample initial solutions, and vector is updated during runtime.
        """

        self.SIZE_GEN = size_gen
        self.MAX_ITER = max_iter
        self.alpha = alpha
        self.vector = vector

        self.variables = list(vector.columns)

        self.best_mae_global = 999999999999

        # self.DEAD_ITER must be fewer than MAX_ITER
        if dead_iter > max_iter:
            raise Exception('ERROR setting DEAD_ITER. The dead iterations must be fewer than the maximum iterations')
        else:
            self.DEAD_ITER = dead_iter

        self.truncation_length = int(size_gen * alpha)

        # initialization of generation
        mus = self.vector[self.variables].loc['mu'].to_list()
        stds = self.vector[self.variables].loc['std'].to_list()
        self.generation = pd.DataFrame(np.random.normal(mus, stds, [self.SIZE_GEN, len(self.variables)]),
                                       columns=self.variables, dtype='float_')

        self.std_bound = std_bound
        self.elite_factor = elite_factor

    # build a generation of size SIZE_GEN from prob vector
    def new_generation(self):
        """
        Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """

        mus = self.vector[self.variables].loc['mu'].to_list()
        stds = self.vector[self.variables].loc['std'].to_list()
        gen = pd.DataFrame(np.random.normal(mus, stds, [self.SIZE_GEN, len(self.variables)]),
                           columns=self.variables, dtype='float_')

        self.generation = self.generation.nsmallest(int(self.elite_factor * len(self.generation)), 'cost')
        self.generation = self.generation.append(gen).reset_index(drop=True)

    # truncate the generation at alpha percent
    def truncation(self):
        """
        Selection of the best individuals of the actual generation.
        """

        self.generation = self.generation.nsmallest(self.truncation_length, 'cost')

    # check each individual of the generation
    def check_generation(self, objective_function):
        """
        Check the cost of each individual in the cost function implemented by the user, and updates the
        generation DataFrame.
        """

        self.generation['cost'] = self.generation.apply(lambda row: objective_function(row[self.variables].to_list()),
                                                        axis=1)

    # update the probability vector
    def update_vector(self):
        """
        From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions.
        """

        for var in self.variables:
            self.vector.at['mu', var], self.vector.at['std', var] = norm.fit(self.generation[var].values)
            if self.vector.at['std', var] < self.std_bound:
                self.vector.at['std', var] = self.std_bound

    # run the class to find the optimum
    def optimize(self, cost_function, output=True):
        """
        Run method to execute the EDA algorithm over a cost function.

        Parameters
        -------------

        cost_function : callable
            Cost function to be optimized which accepts a list of values as argument (values to be optimized).
        output : boolean
            True if desired to print information during runtime. False otherwise.

        Returns
        ------------

        list : List of optimum values found after optimization
        float : Cost of the optimum solution found
        int : Number of iterations until convergence
        """

        not_better = 0
        for i in range(self.MAX_ITER):
            self.check_generation(cost_function)
            self.truncation()
            self.update_vector()

            best_mae_local = self.generation['cost'].min()

            self.history.append(best_mae_local)
            best_ind_local = self.generation[self.generation['cost'] == best_mae_local]

            # update the best values ever
            if best_mae_local < self.best_mae_global:
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0
            else:
                not_better += 1
                if not_better == self.DEAD_ITER:
                    return self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1], self.best_mae_global, \
                           len(self.history)

            self.new_generation()

            if output:
                print('IT ', i, 'best cost ', self.best_mae_global)

        return self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1], self.best_mae_global, len(self.history)
