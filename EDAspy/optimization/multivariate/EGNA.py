#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pybnesian import GaussianNetwork
import pandas as pd


class EGNA:
    """
    Estimation of Gaussian Networks Algorithm. This type of Estimation-of-Distribution Algorithm uses
    a Gaussian Bayesian Network from where new solutions are sampled. This multivariate probabilistic
    model is updated in each iteration with the best individuals of the previous generation.

    EGNA [1] has shown to improve the results for more complex optimization problem compared to the
    univariate EDAs that can be found implemented in this package. Different modifications have been
    done into this algorithm such as in [2] where some evidences are input to the Gaussian Bayesian
    Network in order to restrict the search space in the landscape.

    Example:

        This example uses some very well-known benchmarks from CEC14 conference to be solved using
        an Estimation of Gaussian Networks Algorithm (EGNA).

        .. code-block:: python

            from EDAspy.optimization import EGNA
            from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

            benchmarking = ContinuousBenchmarkingCEC14(10)

            egna = EGNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10,
                        landscape_bounds=(-60, 60))

            best_sol, best_cost, cost_evals = egna.minimize(benchmarking.cec14_4, True)

    References:

        [1]: Larrañaga, P., & Lozano, J. A. (Eds.). (2001). Estimation of distribution algorithms:
        A new tool for evolutionary computation (Vol. 2). Springer Science & Business Media.

        [2]: Vicente P. Soloviev, Pedro Larrañaga and Concha Bielza (2022). Estimation of distribution
        algorithms using Gaussian Bayesian networks to solve industrial optimization problems constrained
        by environment variables. Journal of Combinatorial Optimization.
    """

    best_global_cost = 9999999999
    best_global_ind = np.array(0)

    history = []
    evaluations = []

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 landscape_bounds: tuple,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True):
        r"""
        Args:
            size_gen: Population size. Number of individuals in each generation.
            max_iter: Maximum number of iterations during runtime.
            dead_iter: Stopping criteria. Number of iterations with no improvement after which, the algorithm finish.
            n_variables: Number of variables to be optimized.
            landscape_bounds: Landscape bounds. Limits in the search space.
            alpha: Percentage of population selected to update the probabilistic model.
            elite_factor: Percentage of previous population selected to add to new generation (elite approach).
            disp: Set to True to print convergence messages.
        """

        self.max_it = max_iter
        self.size_gen = size_gen

        assert dead_iter <= max_iter, "dead_it must be lower than max_it"
        self.dead_iter = dead_iter

        self.trunc_size = int(size_gen*alpha)
        self.n_variables = n_variables
        self.elite_factor = elite_factor
        self.disp = disp
        self.landscape_bounds = landscape_bounds
        self.vars = [str(num) for num in range(n_variables)]

        self.pm = GaussianNetwork(self.vars)
        self._initialization()

    def _initialization(self):
        self.generation = np.random.randint(self.landscape_bounds[0], self.landscape_bounds[1],
                                            (self.size_gen, self.n_variables)).astype(float)

    def _evaluation(self, objective_function):
        self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)

    def _truncation(self):
        best_indices = self.evaluations.argsort()[: self.trunc_size]
        self.generation = self.generation[best_indices, :]
        self.evaluations = np.take(self.evaluations, best_indices)

    def _update_pm(self):
        self.pm = GaussianNetwork(self.vars)
        self.pm.fit(pd.DataFrame(self.generation))

    def _new_generation(self):
        self.generation = self.pm.sample(self.size_gen).to_pandas()
        self.generation = self.generation[self.vars].to_numpy()

    def minimize(self, cost_function: callable, output_runtime: bool = True):
        r"""
        Args:
            cost_function: Cost function to be optimized and accepts an array as argument.
            output_runtime: True if information during runtime is desired.
        """
        no_improvement = 0

        for _ in range(self.max_it):
            if no_improvement == self.dead_iter:
                break

            self._evaluation(cost_function)
            self._truncation()
            self._update_pm()

            best_local_cost = self.evaluations[0]
            if best_local_cost < self.best_global_cost:
                self.best_global_cost = best_local_cost
                self.best_global_ind = self.generation[0]
                no_improvement = 0
            else:
                no_improvement += 1

            if output_runtime:
                print('IT: ', _, '\tBest cost: ', self.best_global_cost)

            self._new_generation()
            self.history.append(best_local_cost)

        if self.disp:
            print("\tNFVALS = " + str(len(self.history) * self.size_gen) + " F = " + str(self.best_global_cost))
            print("\tX = " + str(self.best_global_ind))

        return self.best_global_ind, self.best_global_cost, len(self.history) * self.size_gen
