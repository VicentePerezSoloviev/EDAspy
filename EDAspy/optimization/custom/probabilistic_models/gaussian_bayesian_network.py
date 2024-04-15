#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pybnesian import GaussianNetwork, hc, GaussianNetworkType
from ._probabilistic_model import ProbabilisticModel
import pandas as pd


def _check_symmetric(a, tol=1e-8) -> bool:
    return np.all(np.abs(a - a.T) < tol)


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

    def __init__(self, variables: list, white_list: list = None, black_list: list = None, evidences: dict = None):
        """
        :param variables: Number of variables
        :param white_list: List of tuples with mandatory arcs in the BN structure
        :param black_list: List of tuples with forbidden arcs in the BN structure
        :param evidences: Dictionary with name of variables as keys and values fixed as values of the dictionary
        """

        super().__init__(variables)

        self.variables = variables
        self.top_order = None
        self.pm = GaussianNetwork(variables)

        self.white_list = white_list
        self.black_list = black_list

        if evidences:
            self.evidences = evidences
            self.sample = self._sample_evidence

            self.vars_1 = list(set(self.variables) - set(self.evidences.keys()))  # not fixed
            self.vars_2 = list(self.evidences.keys())  # evidence
        else:
            self.sample = self._sample_normal

        self.id = 4
        self.fitted = False

    def learn(self, dataset: np.array, *args, **kwargs):
        """
        Learn a Gaussian Bayesian network from the dataset passed as argument.

        :param dataset: dataset from which learn the GBN.
        """
        data = pd.DataFrame(dataset, columns=self.variables)
        self.pm = GaussianNetwork(self.variables)

        if self.white_list and self.black_list:
            self.pm = hc(data, bn_type=GaussianNetworkType(),
                         arc_blacklist=self.black_list, arc_whitelist=self.white_list)
        elif self.white_list:
            self.pm = hc(data, bn_type=GaussianNetworkType(), arc_whitelist=self.white_list)
        elif self.black_list:
            self.pm = hc(data, bn_type=GaussianNetworkType(), arc_blacklist=self.black_list)
        else:
            self.pm = hc(data, bn_type=GaussianNetworkType())

        self.pm.fit(data)
        self.top_order = self.pm.graph().topological_sort()

        self.fitted = True

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process.

        :return: list of arcs between variables
        :rtype: list
        """

        return self.pm.arcs()

    def sample(self, size: int) -> np.array:
        raise Exception('Not implemented function')

    def _sample_evidence(self, size: int, ordered: bool = True) -> np.array:
        mu_infer, cov_infer = self.inference(list(self.evidences.values()), self.vars_2)
        sample = np.random.multivariate_normal(mu_infer, cov_infer, size)

        dataset = pd.DataFrame(sample, columns=self.vars_1)
        for var in self.vars_2:
            dataset[var] = self.evidences[var]

        top_order = self.pm.graph().topological_sort()
        if ordered:
            return dataset[self.variables].to_numpy()

        return dataset[top_order].to_numpy()

    def _sample_normal(self, size: int) -> np.array:
        dataset = self.pm.sample(size, ordered=True).to_pandas()
        dataset = dataset[self.variables].to_numpy()
        return dataset

    def logl(self, data: pd.DataFrame):
        """
        Returns de log-likelihood of some data in the model.

        :param data: dataset to evaluate its likelihood in the model.
        :return: log-likelihood of the instances in the model.
        :rtype: np.array
        """
        return self.pm.logl(data)

    def get_mu(self, var_mus=None) -> np.array:
        """
        Computes the conditional mean of the Gaussians of each node in the GBN.

        :param var_mus: Variables to compute its Gaussian mean. If None, then all the variables are computed.
        :type var_mus: list
        :return: Array with the conditional Gaussian means.
        :rtype: np.array
        """

        top_order = self.pm.graph().topological_sort()
        mu = np.zeros(len(top_order))

        for i in range(len(top_order)):
            node_name = top_order[i]
            parents = self.pm.cpd(node_name).evidence()
            coefs = self.pm.cpd(node_name).beta

            mu[i] = coefs[0]
            for j in range(1, len(coefs)):
                parent_name = parents[j-1]
                mu[i] += coefs[j] * mu[top_order.index(parent_name)]

        if var_mus:
            idx = [top_order.index(i) for i in var_mus]
            return mu[idx]

        return mu

    def get_sigma(self, var_sigma=None) -> np.array:
        """
        Computes the conditional covariance matrix of the model for the variables in the GBN.

        :param var_sigma: Variables to compute its Gaussian mean. If None, then all the variables are computed.
        :type var_sigma: list
        :return: Matrix with the conditional covariance matrix.
        :rtype: np.array
        """

        top_order = self.pm.graph().topological_sort()
        sigmas = np.zeros((len(top_order), len(top_order)))

        # compute diagonal variances
        for i in range(len(top_order)):
            node_name = top_order[i]
            sigmas[i, i] = self.pm.cpd(node_name).variance
            parents = self.pm.cpd(node_name).evidence()
            coefs = self.pm.cpd(node_name).beta

            for j in range(1, len(coefs)):
                parent_name = parents[j - 1]
                idx_parent = top_order.index(parent_name)
                sigmas[i, i] += sigmas[idx_parent, idx_parent] * coefs[j] ** 2

        # compute covariances
        for i in range(len(top_order)):
            for j in range(i + 1, len(top_order)):
                node_name = top_order[j]
                coefs = self.pm.cpd(node_name).beta
                parents = self.pm.cpd(top_order[j]).evidence()

                for k in range(1, len(coefs)):
                    parent_name = parents[k - 1]
                    sigmas[i, j] += coefs[k] * sigmas[i, top_order.index(parent_name)]

                sigmas[j, i] = sigmas[i, j]

        assert _check_symmetric(sigmas), 'ERROR: covariance matrix must be symmetric.'

        if var_sigma:
            idx = [top_order.index(i) for i in var_sigma]
            return sigmas[np.ix_(idx, idx)]

        return sigmas

    def inference(self, evidence, var_names) -> (np.array, np.array):
        """
        Compute the posterior conditional probability distribution conditioned to some given evidences.
        :param evidence: list of values fixed as evidences in the model.
        :type evidence: list
        :param var_names: list of variables measured in the model.
        :type var_names: list
        :return: (posterior mean, posterior covariance matrix)
        :rtype: (np.array, np.array)
        """
        vars_1 = list(set(self.top_order) - set(var_names))
        vars_2 = var_names

        mus = self.get_mu(vars_1 + vars_2)

        mu_1 = mus[:len(vars_1)]
        mu_2 = mus[len(vars_1):]

        sigma = self.get_sigma(vars_1 + vars_2)
        sigma_11 = sigma[:len(vars_1), :len(vars_1)]
        sigma_12 = sigma[:len(vars_1), len(vars_1):]
        sigma_21 = sigma[len(vars_1):, :len(vars_1)]
        sigma_22 = sigma[len(vars_1):, len(vars_1):]

        inv_sigma_22 = np.linalg.pinv(sigma_22)  # pseudo-inverse

        mu_1_2 = mu_1 + np.dot(sigma_12, np.dot(inv_sigma_22, (np.array(evidence) - mu_2)))
        sigma_1_2 = sigma_11 - np.dot(sigma_12, np.dot(inv_sigma_22, sigma_21))

        return mu_1_2, sigma_1_2

    def maximum_a_posteriori(self, evidence, var_names):
        # TODO: test this
        assert self.fitted, "The Bayesian network has not been fitted yet. Please, learn it first."

        mu_map, sigma_map = self.inference(evidence=evidence, var_names=var_names)
        return mu_map
