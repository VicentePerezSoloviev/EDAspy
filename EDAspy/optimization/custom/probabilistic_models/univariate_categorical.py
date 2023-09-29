#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from ._probabilistic_model import ProbabilisticModel


def obtain_probabilities(array) -> dict:
    res = {}
    unique, counts = np.unique(array, return_counts=True)
    for i in range(len(unique)):
        res[unique[i]] = counts[i]/array.size

    '''for label in list(set(labels) - set(unique)):
        res[label] = 0'''

    return res


class UniCategorical(ProbabilisticModel):

    """
    This probabilistic model is discrete and univariate.
    """

    def __init__(self, variables: list):
        super().__init__(variables)

        self.prob_table = {}  # dictionary with variable: {value: prob}

    def learn(self, dataset: np.array, *args, **kwargs):
        """
        Estimates the independent categorical probability distribution for each variable.

        :param dataset: dataset from which learn the probabilistic model.
        """
        for i in range(self.len_variables):
            label = self.variables[i]
            probs = obtain_probabilities(dataset[:, i])
            self.prob_table[label] = probs

    def sample(self, size: int) -> np.array:
        """
        Samples new solutions from the probabilistic model. In each solution, each variable is sampled
        from its respective categorical distribution.

        :param size: number of samplings of the probabilistic model.
        :return: array with the dataset sampled
        :rtype: np.array
        """
        result = pd.DataFrame(columns=self.variables)
        for i in range(self.len_variables):
            label = self.variables[i]
            result[label] = np.random.choice(list(self.prob_table[label].keys()), size=size,
                                             p=list(self.prob_table[label].values())).tolist()

        return result.to_numpy()

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process. Univariate approaches generate no-edged graphs.

        :return: list of arcs between variables
        :rtype: list
        """
        return list()
