import numpy as np
import pandas as pd


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

    def __init__(self, size_gen, max_iter, dead_iter, alpha, vector, std_bound=0.3):
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
            vector : list
                Array with independent probabilities for each variable: P(x_0=1), P(x_1=1), ..., P(x_n=1)
        """

        self.SIZE_GEN = size_gen
        self.MAX_ITER = max_iter
        self.alpha = alpha
        self.vector = vector

        self.variables = list(range(len(vector)))
        self.best_mae_global = 999999999999

        # self.DEAD_ITER must be fewer than MAX_ITER
        if dead_iter > max_iter:
            raise Exception('ERROR setting DEAD_ITER. The dead iterations must be fewer than the maximum iterations')
        else:
            self.DEAD_ITER = dead_iter

        self.truncation_length = int(size_gen * alpha)

        # initialization of generation
        self.generation = np.random.uniform(low=0, high=1, size=(self.SIZE_GEN, len(self.variables)))
        self.generation = pd.DataFrame(self.generation < [self.vector] * self.SIZE_GEN,
                                       dtype=int, columns=self.variables)

        self.std_bound = std_bound

    # new generation
    def new_generation(self):
        """Build a new generation sampled from the vector of probabilities and updates the generation pandas dataframe
        """
        self.generation = np.random.uniform(low=0, high=1, size=(self.SIZE_GEN, len(self.variables)))
        self.generation = pd.DataFrame(self.generation < [self.vector] * self.SIZE_GEN,
                                       dtype=int, columns=self.variables)

    # check the cost of each individual of the generation
    def check_generation(self, objective_function):
        """Check the cost of each individual in the cost function implemented by the user
        """

        self.generation['cost'] = self.generation.apply(lambda row: objective_function(row[self.variables].to_list()),
                                                        axis=1)

    # truncate the generation at alpha percent
    def truncation(self):
        """
        Selection of the best individuals of the actual generation.
        """

        self.generation = self.generation.nsmallest(self.truncation_length, 'cost')

    # based on the best individuals of the selection, rebuild the prob vector
    def update_vector(self):
        """From the best individuals update the vector of probabilities in order to the next generation can sample from
         it and update the vector of probabilities
        """

        self.vector = np.array(self.generation.drop('cost', axis=1).sum()) / len(self.generation)
        for i in range(len(self.vector)):
            if self.vector[i] < self.std_bound:
                self.vector[i] = self.std_bound

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
