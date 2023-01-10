#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ._probabilistic_model import ProbabilisticModel


class UniBin(ProbabilisticModel):

    """
    This is the simplest probabilistic model implemented in this package. This is used for binary EDAs where
    all the solutions are binary. The implementation involves a vector of independent probabilities [0, 1].
    When sampling, a random float is sampled [0, 1]. If the float is below the probability, then the sampling
    is a 1. Thus, the probabilities show probabilities of a sampling being 1.
    """

    def __init__(self, variables: list, upper_bound: float, lower_bound: float):
        super().__init__(variables)

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.pm = np.zeros(self.len_variables)

        self.id = 2

    def sample(self, size: int) -> np.array:
        """
        Samples new solutions from the probabilistic model. In each solution, each variable is sampled
        from its respective binary probability.

        :param size: number of samplings of the probabilistic model.
        :return: array with the dataset sampled.
        :rtype: np.array
        """

        dataset = np.random.random((size, self.len_variables))
        dataset = dataset < self.pm
        dataset = np.array(dataset, dtype=int)
        return dataset

    def learn(self, dataset: np.array):
        """
        Estimates the independent probability of each variable of being 1.

        :param dataset: dataset from which learn the probabilistic model.
        """

        self.pm = sum(dataset) / len(dataset)
        self.pm[self.pm < self.lower_bound] = self.lower_bound
        self.pm[self.pm > self.upper_bound] = self.upper_bound
