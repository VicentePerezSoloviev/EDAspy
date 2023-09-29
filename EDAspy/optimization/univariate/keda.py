#!/usr/bin/env python
# coding: utf-8
import numpy as np
from typing import Union, List

from ..custom.probabilistic_models import UniKDE
from ..custom.initialization_models import UniformGenInit
from ..eda import EDA


class UnivariateKEDA(EDA):
    """
    Univariate Kernel Density Estimation Algorithm (u_KEDA). New individuals are sampled from a KDE model.
    It can be used for hyper-parameter optimization or to optimize a function.

    u_KEDA [1] is a specific type of Estimation of Distribution Algorithm (EDA) where new individuals
    are sampled from univariate KDEs and are updated in each iteration of the algorithm by the best individuals
    found in the previous iteration. In this implementation each individual is an array of real data so new
    individuals are sampled from a univariate probabilistic model updated in each iteration.

    Example:

        This short example runs UMDAc for a benchmark function optimization problem in the continuous space.

        .. code-block:: python

            from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
            from EDAspy.optimization import UnivariateKEDA

            n_vars = 10
            benchmarking = ContinuousBenchmarkingCEC14(n_vars)

            keda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=10, alpha=0.5,
                                  lower_bound=-100, upper_bound=100)
            # We leave bound by default
            eda_result = keda.minimize(benchmarking.cec4, True)

    References:

        [1]: Larrañaga, P., & Lozano, J. A. (Eds.). (2001). Estimation of distribution algorithms:
        A new tool for evolutionary computation (Vol. 2). Springer Science & Business Media.
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
                 parallelize: bool = False,
                 init_data: np.array = None):
        r"""
        :param size_gen: Population size of each generation.
        :param max_iter: Maximum number of function evaluations.
        :param dead_iter: Stopping criteria. Number of iterations after with no improvement after which EDA stops.
        :param n_variables: Number of variables to be optimized.
        :param alpha: Percentage of population selected to update the probabilistic model.
        :param lower_bound: lower bound for the uniform distribution sampling.
        :param upper_bound: lower bound for the uniform distribution sampling.
        :param elite_factor: Percentage of previous population selected to add to new generation (elite approach).
        :param disp: Set to True to print convergence messages.
        :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
        :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an
        initializer is used.
        :type lower_bound: List of lower bounds of size equal to number of variables OR single bound to all dimensions.
        :type upper_bound: List of upper bounds of size equal to number of variables OR single bound to all dimensions.
        """

        # self.landscape_bounds = landscape_bounds
        self.names_vars = list(range(n_variables))

        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter, n_variables=n_variables,
                         alpha=alpha, elite_factor=elite_factor, disp=disp, parallelize=parallelize,
                         init_data=init_data)

        self.init = UniformGenInit(n_variables=n_variables, lower_bound=lower_bound, upper_bound=upper_bound)
        self.pm = UniKDE(self.names_vars)
