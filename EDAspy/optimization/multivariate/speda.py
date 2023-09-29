#!/usr/bin/env python
# coding: utf-8

from ..eda import EDA
from ..custom.probabilistic_models import SPBN
from ..custom.initialization_models import UniformGenInit

import numpy as np
from typing import Union, List


class SPEDA(EDA):
    """
    Semiparametric Estimation of Distribution Algorithm [1]. This type of Estimation-of-Distribution
    Algorithm uses a semiparametric Bayesian network [2] which allows dependencies between variables
    which have been estimated using KDE with variables which fits a Gaussian distribution. By this
    way, it avoid the assumption of Gaussianity in the variables of the optimization problem. This
    multivariate probabilistic model is updated in each iteration with the best individuals of the
    previous generations.

    SPEDA has shown to improve the results for more complex optimization problem compared to the
    univariate EDAs that can be found implemented in this package, multivariate EDAs such as
    EGNA, or EMNA, and other population-based algorithms. See [1] for numerical results.

    Example:

        This example uses some very well-known benchmarks from CEC14 conference to be solved using
        a Semiparametric Estimation of Distribution Algorithm (SPEDA).

        .. code-block:: python

            from EDAspy.optimization import SPEDA
            from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

            benchmarking = ContinuousBenchmarkingCEC14(10)

            speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-100,
                          upper_bound=100, l=10)

            eda_result = speda.minimize(benchmarking.cec14_4, True)

    References:

        [1]: Vicente P. Soloviev, Concha Bielza and Pedro Larrañaga. Semiparametric Estimation
        of Distribution Algorithm for continuous optimization. 2022

        [2]: Atienza, D., Bielza, C., & Larrañaga, P. (2022). PyBNesian: an extensible Python package
        for Bayesian networks. Neurocomputing, 504, 204-209.

    """

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 lower_bound: Union[np.array, List[float], float],
                 upper_bound: Union[np.array, List[float], float],
                 l: float,
                 alpha: float = 0.5,
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
        :param l: SPEDA is an archive-base approach. Thus, in each generation updates the probabilistic model with
        the best solutions of the previous l generations.
        :param alpha: Percentage of population selected to update the probabilistic model in each generation.
        :param disp: Set to True to print convergence messages.
        :param black_list: list of tuples with the forbidden arcs in the SPBN during runtime.
        :param white_list: list of tuples with the mandatory arcs in the SPBN during runtime.
        :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
        :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an
        initializer is used.
        :type lower_bound: List of lower bounds of size equal to number of variables OR single bound to all dimensions.
        :type upper_bound: List of upper bounds of size equal to number of variables OR single bound to all dimensions.
        """

        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, alpha=alpha, elite_factor=alpha, disp=disp,
                         parallelize=parallelize, init_data=init_data)

        self.vars = [str(i) for i in range(n_variables)]
        # self.landscape_bounds = landscape_bounds
        self.pm = SPBN(self.vars, black_list=black_list, white_list=white_list)

        self.l_len = l*int(size_gen*self.alpha)  # maximum number of individuals in the archive
        self.archive = np.empty((0, self.n_variables))

        # In this implementation the individuals of the first generation are sampled from a uniform distribution
        # to not skew the following estimation of distributions.

        self.init = UniformGenInit(self.n_variables, lower_bound=lower_bound, upper_bound=upper_bound)

    def _update_archive(self):
        self.archive = np.append(self.archive, self.elite_temp, axis=0)
        self.archive = self.archive[-self.l_len:]

    def _update_pm(self):
        """
        Learn the probabilistic model from the best individuals of previous generation, using the best solutions
        of the previous l generations
        """
        self._update_archive()
        self.pm.learn(dataset=self.archive)

    def _new_generation(self):
        # self.generation = np.concatenate([self.pm.sample(size=self.size_gen), [self.best_ind_global]])
        self.generation = self.pm.sample(size=self.size_gen)
        # as it is not an elitist approach we just add the best individual to show always an improvement in the
        # history of the best solution costs
