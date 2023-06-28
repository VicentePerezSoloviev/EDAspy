#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import gaussian_kde
from ._probabilistic_model import ProbabilisticModel


class UniKDE(ProbabilisticModel):

    """
    This class implements the univariate Kernel Density Estimation. With this implementation we are updating N
    univariate KDE in each iteration. When a dataset is given, each column is updated independently.
    """

    def __init__(self, variables: list):
        super().__init__(variables)
        self.kernel = None

        self.id = 7

    def sample(self, size: int) -> np.array:
        """
        Samples new solutions from the probabilistic model. In each solution, each variable is sampled
        from its respective normal distribution.

        :param size: number of samplings of the probabilistic model.
        :return: array with the dataset sampled
        :rtype: np.array
        """
        return self.kernel.resample(size).T

    def learn(self, dataset: np.array, *args, **kwargs):
        """
        Estimates the independent KDE for each variable.

        :param dataset: dataset from which learn the probabilistic model.
        """
        self.kernel = gaussian_kde(dataset.T)

    def print_structure(self) -> list:
        return list()
