#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .multivariate_eda import MultivariateEda


class EMNA(MultivariateEda):
    """
    Estimation of Multivariate Normal Algorithm (EMNA) [1] is a multivariate continuous EDA in which no
    probabilistic graphical models are used during runtime. In each iteration the  new solutions are
    sampled from a multivariate normal distribution built from the elite selection of the previous
    generation.

    In this implementation, as in EGNA, the algorithm is initialized from a uniform sampling in the
    landscape bound you input in the constructor of the algorithm. If a different initialization is
    desired, then you can override the class and this specific method.

    This algorithm is widely used in the literature and compared for different optimization tasks with
    its competitors in the EDAs multivariate continuous research topic.

    Example:

        This example uses some very well-known benchmarks from CEC14 conference to be solved using
        an Estimation of Multivariate Normal Algorithm (EMNA).

        .. code-block:: python

            from EDAspy.optimization import EMNA
            from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

            benchmarking = ContinuousBenchmarkingCEC14(10)

            emna = EMNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10,
                        landscape_bounds=(-60, 60), std_bound=5)

            best_sol, best_cost, n_f_evals = emna.minimize(cost_function=benchmarking.cec14_4)

    References:

        [1]: Larrañaga, P., & Lozano, J. A. (Eds.). (2001). Estimation of distribution algorithms:
        A new tool for evolutionary computation (Vol. 2). Springer Science & Business Media.
    """

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 landscape_bounds: tuple,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 std_bound: float = 0.5,
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
            std_bound: Lower bound imposed in std of the variables to not converge to std=0.
            disp: Set to True to print convergence messages.
        """
        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, landscape_bounds=landscape_bounds,
                         alpha=alpha, elite_factor=elite_factor, disp=disp)

        self.std_bound = std_bound
        self.vector = np.zeros((2, n_variables))

        self._initialization()

    def _update_pm(self):
        for i in range(self.n_variables):
            self.vector[0, i] = np.mean(self.generation[:, i])
            self.vector[1, i] = np.std(self.generation[:, i])
            if self.vector[1, i] < self.std_bound:
                self.vector[1, i] = self.std_bound

    def _new_generation(self):
        gen = np.random.multivariate_normal(self.vector[0, :], self.vector[1, :], [self.size_gen, self.n_variables])

        self.generation = self.generation[: int(self.elite_factor * len(self.generation))]
        self.generation = np.vstack((self.generation, gen))
