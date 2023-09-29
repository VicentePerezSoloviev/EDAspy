#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import Union, List

from ..eda import EDA
from ..custom.probabilistic_models import UniCategorical
from ..custom.initialization_models import CategoricalSampling


class UMDAcat(EDA):
    """
    Categorical version of UMDA algorithm. Each variable approximated an independent probability
    distribution where each variables can have more than two possible values (otherwise better to
    use binary version of UMDA).

    Example:

        This example uses some uses a toy example to show how to use the UMDAcat implementation.

        .. code-block:: python

            from EDAspy.optimization import UMDAcat

            def categorical_cost_function(solution: np.array):
                cost_dict = {
                    'Color': {'Red': 0.1, 'Green': 0.5, 'Blue': 0.3},
                    'Shape': {'Circle': 0.3, 'Square': 0.2, 'Triangle': 0.4},
                    'Size': {'Small': 0.4, 'Medium': 0.2, 'Large': 0.1}
                }
                keys = list(cost_dict.keys())
                choices = {keys[i]: solution[i] for i in range(len(solution))}

                total_cost = 0.0
                for variable, choice in choices.items():
                    total_cost += cost_dict[variable][choice]

                return total_cost

            variables = ['Color', 'Shape', 'Size']
            possible_values = np.array([
                ['Red', 'Green', 'Blue'],
                ['Circle', 'Square', 'Triangle'],
                ['Small', 'Medium', 'Large']], dtype=object
            )

            frequency = np.array([
                [.33, .33, .33],
                [.33, .33, .33],
                [.33, .33, .33]], dtype=object
            )

            n_variables = len(variables)

            umda_cat = UMDAcat(size_gen=10, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5,
                               frequency=frequency, possible_values=possible_values)

            umda_cat_result = umda_cat.minimize(categorical_cost_function, True)
        """

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 possible_values: Union[List, np.array],
                 frequency: Union[List, np.array],
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: np.array = None):
        r"""
        :param size_gen: Population size. Number of individuals in each generation.
        :param max_iter: Maximum number of iterations during runtime.
        :param dead_iter: Stopping criteria. Number of iterations with no improvement after which, the algorithm finish.
        :param n_variables: Number of variables to be optimized.
        :param possible_values: 2D structure where each row represents the possible values that can have each dimension.
        :param frequency: 2D structure with same size as possible_values and represent the frequency of each element.
        :param alpha: Percentage of population selected to update the probabilistic model.
        :param elite_factor: Percentage of previous population selected to add to new generation (elite approach).
        :param disp: Set to True to print convergence messages.
        :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
        :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an
        initializer is used.
        """

        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, alpha=alpha, elite_factor=elite_factor, disp=disp,
                         parallelize=parallelize, init_data=init_data)

        self.vars = [str(i) for i in range(n_variables)]
        # self.landscape_bounds = landscape_bounds
        self.pm = UniCategorical(self.vars)
        self.init = CategoricalSampling(self.n_variables, possible_values=possible_values, frequency=frequency)
