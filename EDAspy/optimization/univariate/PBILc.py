#!/usr/bin/env python
# coding: utf-8

import numpy as np


class PBILc:
    """
    Population-based Incremental Learning algorithm continuous (PBILc) [1]. This is a generalization of
    the Univariate Marginal Distribution Algorithms also implemented in EDAspy. In this case, for the
    continuous space we use normal distributions that do not only consider the best individuals of
    previous generation but also the best, second best and worst of the previous one. This approach
    is motivated by the PBIL in discrete spaces [2] and the Differential Evolution.

    We can also set a lower bound to avoid the algorithm to reach to low standard deviations for the
    variables. This value is preset but can be modified by the user.

    This algorithm is widely used in the state-of-the-art and traditionally compared with the
    performance of other algorithms such as the Compact Genetic Algorithm or the concrete case of UMDA.

    Example:

        This example uses some very well-known benchmarks from CEC14 conference to be solved using
        a Population-based Incremental Learning Algorithm (PBIL) in the continuous space.

        .. code-block:: python

            from EDAspy.optimization import PBILc
            from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

            n_vars = 10
            benchmarking = ContinuousBenchmarkingCEC14(n_vars)

            pbil = PBILc(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_vars)
            best_sol, best_cost, n_f_evals = pbil.minimize(cost_function=benchmarking.cec14_4)

    References:

        [1]: Sebag, M., & Ducoulombier, A. (1998, September). Extending population-based incremental
        learning to continuous search spaces. In International Conference on Parallel Problem Solving
        from Nature (pp. 418-427). Springer, Berlin, Heidelberg.

        [2]: Baluja, S. (1994). Population-based incremental learning. a method for integrating genetic
        search based function optimization and competitive learning. Carnegie-Mellon Univ Pittsburgh
        Pa Dept Of Computer Science.
    """

    history = []
    evaluations = []

    best_mae_global = 99999999999
    best_ind_global = np.array(0)

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 learning_rate: float = 0.5,
                 vector: np.array = None,
                 std_bound: float = 0.3,
                 elite_factor: float = 0.4,
                 disp: bool = True):

        self.size_gen = size_gen
        self.max_iter = max_iter
        self.dead_iter = dead_iter
        self.n_variables = n_variables
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.truncation_length = int(size_gen * alpha)
        self.std_bound = std_bound
        self.elite_factor = elite_factor
        self.disp = disp

        if vector is not None:
            assert vector.shape == (2, self.n_variables)
            self.vector = vector
        else:
            self.vector = self._initialize_vector()

        # initialization of generation
        self.generation = np.random.normal(
            self.vector[0, :], self.vector[1, :], [self.size_gen, self.n_variables]
        )

    def _initialize_vector(self):
        vector = np.zeros((2, self.n_variables))

        vector[0, :] = np.pi  # mu
        vector[1, :] = 0.5  # std

        return vector

    # build a generation of size SIZE_GEN from prob vector
    def _new_generation(self):
        gen = np.random.normal(
            self.vector[0, :], self.vector[1, :], [self.size_gen, self.n_variables]
        )

        self.generation = self.generation[: int(self.elite_factor * len(self.generation))]
        self.generation = np.vstack((self.generation, gen))

    # check each individual of the generation
    def _check_generation(self, objective_function):
        """
        Check the cost of each individual in the cost function implemented by the user, and updates the
        generation DataFrame.
        """
        self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)

    # truncate the generation at alpha percent
    def _truncation(self):
        """
        Selection of the best individuals of the actual generation.
        """
        best_indices = self.evaluations.argsort()[: self.truncation_length]
        self.generation = self.generation[best_indices, :]
        self.evaluations = np.take(self.evaluations, best_indices)

    def _update_pm(self):
        for i in range(self.n_variables):
            self.vector[0, i] = (1 - self.learning_rate) * np.mean(self.generation[:, i])
            self.vector[0, i] += self.learning_rate * np.mean(self.generation[0, i] +
                                                              self.generation[0, i] + self.generation[-1, i])
            self.vector[1, i] = np.std(self.generation)

            if self.vector[1, i] < self.std_bound:
                self.vector[1, i] = self.std_bound

    # run the class to find the optimum
    def minimize(self, cost_function: callable, output_runtime: bool = True):
        r"""
        Args:
            cost_function: Cost function to be optimized and accepts an array as argument.
            output_runtime: True if information during runtime is desired.
        """

        self.history = []
        not_better = 0

        for _ in range(self.max_iter):
            self._check_generation(cost_function)
            # self._truncation()
            self._update_pm()

            best_mae_local = min(self.evaluations)

            self.history.append(best_mae_local)
            best_ind_local = np.where(self.evaluations == best_mae_local)[0][0]
            best_ind_local = self.generation[best_ind_local]

            # update the best values ever
            if best_mae_local < self.best_mae_global:
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0

            else:
                not_better += 1
                if not_better == self.dead_iter:
                    break

            self._new_generation()

            if output_runtime:
                print('IT: ', _, '\tBest cost: ', self.best_mae_global)

        if self.disp:
            print("\tNFVALS = " + str(len(self.history) * self.size_gen) + " F = " + str(self.best_mae_global))
            print("\tX = " + str(self.best_ind_global))

        return self.best_ind_global, self.best_mae_global, len(self.history) * self.size_gen
