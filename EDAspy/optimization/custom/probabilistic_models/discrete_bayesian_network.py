#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator

from ._probabilistic_model import ProbabilisticModel


class BN(ProbabilisticModel):

    """
    This probabilistic model is Discrete Bayesian Network. This implementation uses pgmpy library [1].

    References:
        [1]: Ankan, A., & Panda, A. (2015). pgmpy: Probabilistic graphical models using python. In
        Proceedings of the 14th python in science conference (scipy 2015) (Vol. 10). Citeseer.
    """

    def __init__(self, variables: list):
        """
        :param variables: Number of variables.
        """
        # Future implementation: add white and black lists, and evidences

        super().__init__(variables)

        self.variables = variables
        self.pm = BayesianNetwork()

        self.id = 7

    def learn(self, dataset: np.array, *args, **kwargs):
        """
        Learn a discrete Bayesian network from the dataset passed as argument.

        :param dataset: dataset from which learn the GBN.
        """
        data = pd.DataFrame(dataset, columns=self.variables, dtype="category")

        # initialize model
        self.pm = BayesianNetwork()

        # add nodes
        self.pm.add_nodes_from(self.variables)

        # learn structure
        es = HillClimbSearch(data)
        best_structure = es.estimate(scoring_method='bicscore', max_iter=1000, show_progress=False)

        for edge in best_structure.edges():
            self.pm.add_edge(edge[0], edge[1])

        # fit model
        self.pm.fit(data, estimator=MaximumLikelihoodEstimator)

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process.

        :return: list of arcs between variables
        :rtype: list
        """
        return self.pm.edges()

    def sample(self, size: int) -> np.array:
        dataset = self.pm.simulate(size, show_progress=False)
        dataset = dataset[self.variables].to_numpy()
        return dataset
