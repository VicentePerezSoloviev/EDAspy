#!/usr/bin/env python
# coding: utf-8
import numpy as np
from typing import Union, List

from ..eda import EDA
from ..custom.probabilistic_models import MultiGauss
from ..custom.initialization_models import UniformGenInit


class EMNA(EDA):
    """
    Estimation of Multivariate Normal Algorithm (EMNA) [1] is a multivariate continuous EDA in which no
    probabilistic graphical models are used during runtime. In each iteration the  new solutions are
    sampled from a multivariate normal distribution built from the elite selection of the previous
    generation.

    In this implementation, as in EGNA, the algorithm is initialized from a uniform sampling in the
    landscape bound you input in the constructor of the algorithm. If a different initialization_models is
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

            emna = EMNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-100,
                        upper_bound=100)

            eda_result = emna.minimize(cost_function=benchmarking.cec14_4)

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
                 lower_factor: float = 0.5,
                 upper_factor: float = 100,
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
        :param lower_factor: Lower bound imposed in std of the variables to not converge to std=0.
        :param upper_factor: Upper bound imposed in std of the variables.
        :param disp: Set to True to print convergence messages.
        :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
        :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an
        initializer is used.
        :type lower_bound: List of lower bounds of size equal to number of variables OR single bound to all dimensions.
        :type upper_bound: List of upper bounds of size equal to number of variables OR single bound to all dimensions.
        """
        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, alpha=alpha, elite_factor=elite_factor, disp=disp,
                         parallelize=parallelize, init_data=init_data)

        self.lower_bound = lower_factor
        self.upper_bound = upper_factor

        self.pm = MultiGauss(list(range(n_variables)), lower_factor, upper_factor)
        self.init = UniformGenInit(self.n_variables, lower_bound=lower_bound, upper_bound=upper_bound)
