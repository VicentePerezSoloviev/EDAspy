#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ._probabilistic_model import ProbabilisticModel


class MultiGauss(ProbabilisticModel):

    """
    This class implements all the code needed to learn and sample multivariate Gaussian distributions defined
    by a vector of means and a covariance matrix among the variables. This is a simpler approach compared to
    Gaussian Bayesian networks, as multivariate Gaussian distributions do not identify conditional dependeces
    between the variables.

    """

    def __init__(self,
                 variables: list,
                 lower_bound: float,
                 upper_bound: float):
        super().__init__(variables)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.pm_means = np.empty(self.len_variables)
        self.pm_cov = np.zeros((self.len_variables, self.len_variables))

        self.id = 3

    def sample(self, size: int) -> np.array:
        """
        Samples the multivariate Gaussian distribution several times defined by the user. The dataset
        is returned as a numpy matrix.

        :param size: number of samplings of the Gaussian Bayesian network.
        :return: array with the dataset sampled.
        :rtype: np.array
        """

        return np.random.multivariate_normal(self.pm_means, self.pm_cov, size)

    def learn(self, dataset: np.array):
        """
        Estimates a multivariate Gaussian distribution from the dataset.

        :param dataset: dataset from which learn the multivariate Gaussian distribution.
        """

        for i in range(self.len_variables):
            self.pm_means[i] = np.mean(dataset[:, i])

        self.pm_cov = np.cov(dataset.T)
        self.pm_cov[self.pm_cov < self.lower_bound] = self.lower_bound
        self.pm_cov[self.pm_cov > self.upper_bound] = self.upper_bound
        np.fill_diagonal(self.pm_cov, dataset.std(0))

