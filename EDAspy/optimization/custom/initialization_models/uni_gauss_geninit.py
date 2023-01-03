#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ..probabilistic_models import UniGauss
from ._generation_init import GenInit


class UniGaussGenInit(GenInit):

    """
    Initial generation simulator based on the probabilistic model of univariate binary probabilities.
    """

    def __init__(self,
                 n_variables: int,
                 means_vector: np.array = np.empty(0),
                 stds_vector: np.array = np.empty(0),
                 lower_bound: int = -100,
                 higher_bound: int = 100):

        """
        :param n_variables: Number of variables
        :param means_vector: Array of means to initialize the item.
        :type means_vector: np.array
        :param stds_vector: Array of standard deviations to initialize the item.
        :type stds_vector: np.array
        :param lower_bound: lower bound for the random stds_vector.
        :param higher_bound: higher bound for the random stds_vector.
        """

        super().__init__(n_variables)

        assert len(means_vector) == len(stds_vector), "Lengths of means and stds vector must be the same."

        if len(means_vector) == 0:
            self.means_vector = np.random.uniform(low=lower_bound, high=higher_bound, size=n_variables)
            # the stds are random but using the lower_bound and higher_bound, minimizing the std
            self.stds_vector = np.random.uniform(low=abs(lower_bound/4), high=abs(higher_bound/2), size=n_variables)
        else:
            self.means_vector = means_vector
            self.stds_vector = stds_vector

        self.pm = UniGauss(list(range(self.n_variables)), lower_bound)
        self.pm.pm = np.array([self.means_vector, self.stds_vector])

        self.id = 1

    def sample(self, size) -> np.array:
        """
        Sample several times the initializator.

        :param size: number of samplings.
        :return: array with the dataset sampled.
        :rtype: np.array.
        """

        return self.pm.sample(size=size)
