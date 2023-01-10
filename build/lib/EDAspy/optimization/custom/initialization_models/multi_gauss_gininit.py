#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ..probabilistic_models import MultiGauss
from ._generation_init import GenInit


class MultiGaussGenInit(GenInit):

    """
    Initial generation simulator based on the probabilistic model of multivariate Gaussian distribution.
    """

    def __init__(self,
                 n_variables: int,
                 means_vector: np.array = np.empty(0),
                 cov_matrix: np.array = np.empty(0),
                 lower_bound: float = -100,
                 upper_bound: float = 100):

        """
        :param n_variables: Number of variables
        :param means_vector: Array of means to initialize the item.
        :type means_vector: np.array
        :param cov_matrix: Covariance matrix to initialize the item.
        :type cov_matrix: np.array
        :param lower_bound: lower bound for the random covariance matrix.
        :param upper_bound: upper bound for the random covariance matrix.
        """

        super().__init__(n_variables)
        assert len(means_vector) == len(cov_matrix), "Lengths of means vector and covariance matrix must be the same."

        if len(means_vector) == 0:
            self.means_vector = np.random.randint(low=lower_bound, high=upper_bound, size=n_variables)
            self.cov_matrix = np.random.randint(low=lower_bound, high=upper_bound, size=(n_variables, n_variables))
        else:
            self.means_vector = means_vector
            self.cov_matrix = cov_matrix

        self.pm = MultiGauss(list(range(n_variables)), lower_bound, upper_bound)
        self.pm.pm_means = self.means_vector
        self.pm.pm_cov = self.cov_matrix

        self.id = 3

    def sample(self, size: int) -> np.array:
        """
        Sample several times the initializator.

        :param size: number of samplings.
        :return: array with the dataset sampled.
        :rtype: np.array
        """

        return self.pm.sample(size)
