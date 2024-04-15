#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ._probabilistic_model import ProbabilisticModel


class AdaptUniGauss(ProbabilisticModel):

    """
    This class implements the adaptive univariate Gaussians. With this implementation we are updating N univariate
    Gaussians in each iteration. When a dataset is given, each column is updated independently. The implementation
    involves a matrix with two rows, in which the first row are the means and the second one, are the standard
    deviations. Each Gaussian mean is updates as follows, where the two best individuals and the worst are considered.

    .. math::
        \\mu_{l+1} = (1 - \\alpha) \\mu_l + \\alpha (x^{best, 1}_l + x^{best, 2}_l - x^{worst}_l)

    """

    def __init__(self, variables: list, lower_bound: float, alpha: float = .5):
        super().__init__(variables)

        self.pm = np.zeros((2, self.len_variables))
        self.lower_bound = lower_bound

        self.alpha = alpha

        self.id = 8

    def sample(self, size: int) -> np.array:
        """
        Samples new solutions from the probabilistic model. In each solution, each variable is sampled
        from its respective normal distribution.

        :param size: number of samplings of the probabilistic model.
        :return: array with the dataset sampled
        :rtype: np.array
        """
        return np.random.normal(
            self.pm[0, :], self.pm[1, :], (size, self.len_variables)
        )

    def learn(self, dataset: np.array, *args, **kwargs):
        """
        Estimates the independent Gaussian for each variable.

        :param dataset: dataset from which learn the probabilistic model.
        :param alpha: adaptive parameter in formula
        """
        for i in range(len(self.variables)):
            # Here we assume that the given dataset is ordered based on the objective to be optimized
            self.pm[0, i] = ((1 - self.alpha) * np.mean(dataset[:, i])) + \
                            self.alpha * (dataset[0, i] + dataset[1, i] - dataset[-1, i])
            self.pm[1, i] = np.std(dataset[:, i])

            if self.pm[1, i] < self.lower_bound:
                self.pm[1, i] = self.lower_bound

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process. Univariate approaches generate no-edged graphs.

        :return: list of arcs between variables
        :rtype: list
        """
        return list()
