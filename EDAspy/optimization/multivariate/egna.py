#!/usr/bin/env python
# coding: utf-8

from pybnesian import GaussianNetwork
import pandas as pd
from .multivariate_eda import MultivariateEda


class EGNA(MultivariateEda):
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
        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, landscape_bounds=landscape_bounds,
                         alpha=alpha, elite_factor=elite_factor, disp=disp)

        self.pm = GaussianNetwork(self.vars)

    def _update_pm(self):
        self.pm = GaussianNetwork(self.vars)
        self.pm.fit(pd.DataFrame(self.generation))

    def _new_generation(self):
        self.generation = self.pm.sample(self.size_gen).to_pandas()
        self.generation = self.generation[self.vars].to_numpy()
