#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .probabilistic_model import ProbabilisticModel


class MultiGauss(ProbabilisticModel):

    def __init__(self, variables, lower_bound: float, upper_bound: float):
        super().__init__(variables)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.pm_means = np.empty(self.len_variables)
        self.pm_cov = np.zeros((self.len_variables, self.len_variables))

        self.id = 3

    def sample(self, size: int):
        dataset = np.random.multivariate_normal(self._pm_means, self.pm_cov, size)
        return dataset

    def learn(self, dataset: np.array):
        for i in range(self.len_variables):
            self._pm_means[i] = np.mean(dataset[:, i])

        self.pm_cov = np.cov(dataset.T)
        self.pm_cov[self.pm_cov < self.lower_bound] = self.lower_bound
        self.pm_cov[self.pm_cov > self.upper_bound] = self.upper_bound
        np.fill_diagonal(self.pm_cov, dataset.std(0))

    def export_settings(self):
        return self.id, self.lower_bound, self.upper_bound

    @property
    def pm_means(self) -> np.array:
        return self._pm_means

    @pm_means.setter
    def pm_means(self, value: np.array):
        assert len(value) == len(self.variables), "The argument must be an array of 1 dimension and length the " \
                                                  "number of variables"
        self._pm_means = value

    @property
    def pm_cov(self) -> np.array:
        return self._pm_cov

    @pm_cov.setter
    def pm_cov(self, value: np.array):
        len_vars = len(self.variables)
        assert value.shape == (len_vars, len_vars), "The argument must be an square matrix of size number of variables"
        self._pm_cov = value
