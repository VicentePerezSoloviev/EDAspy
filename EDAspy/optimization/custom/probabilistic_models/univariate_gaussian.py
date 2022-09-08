#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .probabilistic_model import ProbabilisticModel


class UniGauss(ProbabilisticModel):

    def __init__(self, variables: list, lower_bound: float):
        super().__init__(variables)

        self.pm = np.zeros((2, self.len_variables))
        self.lower_bound = lower_bound

        self.id = 1

    def sample(self, size: int):
        dataset = np.random.normal(
            self.pm[0, :], self.pm[1, :], [size, self.len_variables]
        )
        return dataset

    def learn(self, dataset: np.array):
        for i in range(len(self.variables)):
            self.pm[0, i] = np.mean(dataset[:, i])
            self.pm[1, i] = np.std(dataset[:, i])

            if self.pm[1, i] < self.lower_bound:
                self.pm[1, i] = self.lower_bound
