#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ._generation_init import GenInit


class UniformGenInit(GenInit):

    """
    Initial generation simulator based on independent uniform distributions.
    """

    def __init__(self,
                 n_variables: int,
                 lower_bound: float = -100,
                 upper_bound: float = 100
                 ):
        """
        :param n_variables: Number of variables.
        :param lower_bound: lower bound for the uniform distribution sampling.
        :param upper_bound: lower bound for the uniform distribution sampling.
        """

        super().__init__(n_variables)

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

        return np.random.randint(self.lower_bound, self.upper_bound, (size, self.n_variables)).astype(float)
