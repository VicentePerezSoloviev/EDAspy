#!/usr/bin/env python
# coding: utf-8
import numpy as np
from typing import List, Union

from ..eda import EDA
from ..custom.probabilistic_models import GBN
from ..custom.initialization_models import UniformGenInit


class EGNA(EDA):
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
                        lower_bound=-100, upper_bound=100)

            eda_result = egna.minimize(benchmarking.cec14_4, True)

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
                 lower_bound: Union[np.array, List[float], float],
                 upper_bound: Union[np.array, List[float], float],
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 black_list: list = None,
                 white_list: list = None,
                 parallelize: bool = False,
                 init_data: np.array = None):
        r"""
        :param size_gen: Population size. Number of individuals in each generation.
        :param max_iter: Maximum number of iterations during runtime.
        :param dead_iter: Stopping criteria. Number of iterations with no improvement after which, the algorithm finish.
        :param n_variables: Number of variables to be optimized.
        :param lower_bound: lower bound for the uniform distribution sampling.
        :param upper_bound: lower bound for the uniform distribution sampling.
        :param alpha: Percentage of population selected to update the probabilistic model.
        :param elite_factor: Percentage of previous population selected to add to new generation (elite approach).
        :param disp: Set to True to print convergence messages.
        :param black_list: list of tuples with the forbidden arcs in the GBN during runtime.
        :param white_list: list of tuples with the mandatory arcs in the GBN during runtime.
        :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
        :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an
        initializer is used.
        :type lower_bound: List of lower bounds of size equal to number of variables OR single bound to all dimensions.
        :type upper_bound: List of upper bounds of size equal to number of variables OR single bound to all dimensions.
        """

        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, alpha=alpha, elite_factor=elite_factor, disp=disp,
                         parallelize=parallelize, init_data=init_data)

        self.vars = [str(i) for i in range(n_variables)]
        # self.landscape_bounds = landscape_bounds
        self.pm = GBN(self.vars, black_list=black_list, white_list=white_list)
        self.init = UniformGenInit(self.n_variables, lower_bound=lower_bound, upper_bound=upper_bound)
