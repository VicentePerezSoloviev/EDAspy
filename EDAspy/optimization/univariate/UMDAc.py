#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import norm
from .UMDA import UMDA


class UMDAc(UMDA):
    """
    Univariate marginal Estimation of Distribution algorithm continuous. New individuals are sampled
    from a univariate normal probabilistic model. It can be used for hyper-parameter optimization
    or to optimize a function.

    UMDA [1] is a specific type of Estimation of Distribution Algorithm (EDA) where new individuals
    are sampled from univariate normal distributions and are updated in each iteration of the
    algorithm by the best individuals found in the previous iteration. In this implementation each
    individual is an array of 0s and 1s so new individuals are sampled from a univariate probabilistic
    model updated in each iteration. Optionally it is possible to set lower bound to the standard
    deviation of the normal distribution for the variables to avoid premature convergence.

    This algorithms has been widely used for different applications such as in [2] where it is
    applied to optimize the parameters of a quantum paremetric circuit and is shown how it outperforms
    other approaches in specific situations.

    Example:

        This short example runs UMDAc for a toy example of the One-Max problem in the continuous space.

        .. code-block:: python

            from EDAspy.benchmarks import one_max
            from EDAspy.optimization import UMDAc, UMDAd

            def one_max_min(array):
                return -one_max(array)

            umda = UMDAc(size_gen=100, max_iter=100, dead_iter=10, n_variables=10, alpha=0.5)
            # We leave bound by default
            best_sol, best_cost, cost_evals = umda.minimize(one_max_min, True)

    References:

        [1]: Larrañaga, P., & Lozano, J. A. (Eds.). (2001). Estimation of distribution algorithms:
        A new tool for evolutionary computation (Vol. 2). Springer Science & Business Media.

        [2]: Vicente P. Soloviev, Pedro Larrañaga and Concha Bielza (2022, July). Quantum Parametric
        Circuit Optimization with Estimation of Distribution Algorithms. In 2022 The Genetic and
        Evolutionary Computation Conference (GECCO). DOI: https://doi.org/10.1145/3520304.3533963
    """

    best_mae_global = 999999999999
    best_ind_global = -1

    history = []
    evaluations = np.array(0)

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 vector: np.array = None,
                 std_bound: float = 0.3,
                 elite_factor: float = 0.4,
                 disp: bool = True):
        r"""
        Args:
            size_gen: Population size of each generation.
            max_iter: Maximum number of function evaluations.
            dead_iter: Stopping criteria. Number of iterations after with no improvement after which EDA stops.
            n_variables: Number of variables to be optimized.
            alpha: Percentage of population selected to update the probabilistic model.
            vector: Array with shape (2, n_variables) where rows are mean and std of the parameters to be optimized.
            std_bound: Lower bound imposed in std of the variables to not converge to std=0.
            elite_factor: Percentage of previous population selected to add to new generation (elite approach).
            disp: Set to True to print convergence messages.
        """

        super().__init__(size_gen, max_iter, dead_iter, n_variables, alpha, elite_factor, disp)
        self.std_bound = std_bound

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
        """
        Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """
        gen = np.random.normal(
            self.vector[0, :], self.vector[1, :], [self.size_gen, self.n_variables]
        )

        self.generation = self.generation[: int(self.elite_factor * len(self.generation))]
        self.generation = np.vstack((self.generation, gen))

    # update the probability vector
    def _update_vector(self):
        """
        From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions.
        """
        for i in range(self.n_variables):
            self.vector[0, i], self.vector[1, i] = norm.fit(self.generation[:, i])
            if self.vector[1, i] < self.std_bound:
                self.vector[1, i] = self.std_bound
