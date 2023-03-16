#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ..probabilistic_models import UniBin
from ._generation_init import GenInit


class UniBinGenInit(GenInit):

    """
    Initial generation simulator based on the probabilistic model of univariate binary probabilities.
    """

    def __init__(self, n_variables: int, means_vector: np.array = np.empty(0)):
        """
        :param n_variables: Number of variables.
        :param means_vector: Array of means to initialize the item.
        :type means_vector: np.array
        """
        super().__init__(n_variables)

        if len(means_vector) == 0:
            self.means_vector = np.array([0.5] * self.n_variables)
        else:
            self.means_vector = means_vector

        self.pm = UniBin(list(range(self.n_variables)), lower_bound=0, upper_bound=1)  # dismiss bounds
        self.pm.pm = self.means_vector

        self.id = 2

    def sample(self, size: int) -> np.array:
        """
        Sample several times the initializator.

        :param size: number of samplings.
        :return: array with the dataset sampled
        :rtype: np.array
        """

        return self.pm.sample(size)
