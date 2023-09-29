#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ._generation_init import GenInit
from typing import Union, List


class UniformGenInit(GenInit):

    """
    Initial generation simulator based on independent uniform distributions.
    """

    def __init__(self,
                 n_variables: int,
                 lower_bound: Union[np.array, List[float], float] = -100,
                 upper_bound: Union[List[float], float] = 100
                 ):
        """
        :param n_variables: Number of variables.
        :param lower_bound: lower bound for the uniform distribution sampling.
        :param upper_bound: lower bound for the uniform distribution sampling.
        :rtype lower_bound: List of lower bounds of size equal to number of variables OR single bound to all dimensions.
        :rtype upper_bound: List of upper bounds of size equal to number of variables OR single bound to all dimensions.
        """

        super().__init__(n_variables)

        if (type(lower_bound) is np.array) or (type(lower_bound) is list):
            assert len(lower_bound) == n_variables, "Number of lower bounds does not match the dimension size."

        if (type(upper_bound) is np.array) or (type(upper_bound) is list):
            assert len(upper_bound) == n_variables, "Number of upper bounds does not match the dimension size."

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.id = 4

    def sample(self, size: int) -> np.array:
        """
        Sample several times the initializator.

        :param size: number of samplings.
        :return: array with the dataset sampled.
        :rtype: np.array.
        """

        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.n_variables))
