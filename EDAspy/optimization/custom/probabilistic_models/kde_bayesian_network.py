#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pybnesian import KDENetwork, hc
from ._probabilistic_model import ProbabilisticModel
import pandas as pd


class KDEBN(ProbabilisticModel):

    """
    This probabilistic model is a Kernel Density Estimation Bayesian network [1]. It allows dependencies
    between variables which have been estimated using KDE.

    References:

        [1]: Atienza, D., Bielza, C., & Larrañaga, P. (2022). PyBNesian: an extensible Python package
        for Bayesian networks. Neurocomputing, 504, 204-209.
    """

    def __init__(self, variables: list, white_list: list = None, black_list: list = None):
        """
        :param variables: Number of variables
        :param white_list: List of tuples with mandatory arcs in the BN structure
        :param black_list: List of tuples with forbidden arcs in the BN structure
        """

        super().__init__(variables)

        self.variables = variables
        self.pm = KDENetwork(variables)

        self.white_list = white_list
        self.black_list = black_list

        self.id = 6

    def learn(self, dataset: np.array, num_folds: int = 10):
        """
        Learn a KDE Bayesian network from the dataset passed as argument.

        :param dataset: dataset from which learn the KDEBN.
        :param num_folds: Number of folds used for the SPBN learning. The higher, the more accurate, but also higher
        CPU demand. By default, it is set to 10.
        """

        self.pm = KDENetwork(self.variables)

        if self.white_list and self.black_list:
            self.pm = hc(pd.DataFrame(dataset), start=self.pm, operators=["arcs"],
                         arc_whitelist=self.white_list, arc_blacklist=self.black_list, num_folds=num_folds)
        elif self.white_list:
            self.pm = hc(pd.DataFrame(dataset), start=self.pm, operators=["arcs"],
                         arc_whitelist=self.white_list, num_folds=num_folds)
        elif self.black_list:
            self.pm = hc(pd.DataFrame(dataset), start=self.pm, operators=["arcs"],
                         arc_blacklist=self.black_list, num_folds=num_folds)
        else:
            self.pm = hc(pd.DataFrame(dataset), start=self.pm, operators=["arcs"], num_folds=num_folds)

        self.pm.fit(pd.DataFrame(dataset))

    def sample(self, size: int) -> np.array:
        """
        Samples the KDE Bayesian network several times defined by the user. The dataset is returned
        as a numpy matrix. The sampling process is implemented using probabilistic logic sampling.

        :param size: number of samplings of the KDE Bayesian network.
        :return: array with the dataset sampled.
        :rtype: np.array
        """

        dataset = self.pm.sample(size).to_pandas()
        dataset = dataset[self.variables].to_numpy()
        return dataset

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process.

        :return: list of arcs between variables
        :rtype: list
        """

        return self.pm.arcs()

    def logl(self, data: pd.DataFrame):
        """
        Returns de log-likelihood of some data in the model.

        :param data: dataset to evaluate its likelihood in the model.
        :return: log-likelihood of the instances in the model.
        :rtype: np.array
        """
        return self.pm.logl(data)
