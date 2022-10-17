#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pybnesian import GaussianNetwork, hc, GaussianNetworkType
from ._probabilistic_model import ProbabilisticModel
import pandas as pd


class GBN(ProbabilisticModel):

    """
    This probabilistic model is  Gaussian Bayesian Network. All the relationships between the variables in
    the model are defined to be linearly Gaussian, and the variables distributions are assumed to be
    Gaussian. This is a very common approach when facing to continuous data as it is relatively easy and fast
    to learn a Gaussian distributions between variables. This implementation uses Pybnesian library [1].

    References:

        [1]: Atienza, D., Bielza, C., & Larrañaga, P. (2022). PyBNesian: an extensible Python package
        for Bayesian networks. Neurocomputing, 504, 204-209.
    """

    def __init__(self, variables: list):

        """
        :param variables: Number of variables
        """

        super().__init__(variables)

        self.variables = variables
        self.pm = GaussianNetwork(variables)

        self.id = 4

    def learn(self, dataset: np.array):
        """
        Learn a Gaussian Bayesian network from the dataset passed as argument.

        :param dataset: dataset from which learn the GBN.
        """

        self.pm = GaussianNetwork(self.variables)
        self.pm = hc(pd.DataFrame(dataset), bn_type=GaussianNetworkType())
        self.pm.fit(pd.DataFrame(dataset))

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process.

        :return: list of arcs between variables
        :rtype: list
        """

        return self.pm.arcs()

    def sample(self, size: int) -> np.array:
        """
        Samples the Gaussian Bayesian network several times defined by the user. The dataset is returned
        as a numpy matrix. The sampling process is implemented using probabilistic logic sampling.

        :param size: number of samplings of the Gaussian Bayesian network.
        :return: array with the dataset sampled.
        :rtype: np.array
        """

        dataset = self.pm.sample(size).to_pandas()
        dataset = dataset[self.variables].to_numpy()
        return dataset
