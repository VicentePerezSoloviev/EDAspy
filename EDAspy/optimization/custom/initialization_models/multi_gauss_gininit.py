#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ..probabilistic_models import MultiGauss
from .generation_init import GenInit


class MultiGaussGenInit(GenInit):

    def __init__(self,
                 n_variables: int,
                 means_vector: np.array = np.empty(0),
                 cov_matrix: np.array = np.empty(0),
                 lower_bound: float = -100,
                 upper_bound: float = 100):

        super().__init__(n_variables)
        assert len(means_vector) == len(cov_matrix), "Lengths of means vector and covariance matrix must be the same."

        if len(means_vector) == 0:
            self.means_vector = np.random.randint(low=lower_bound, high=upper_bound, size=n_variables)
            self.cov_matrix = np.random.randint(low=lower_bound, high=upper_bound, size=(n_variables, n_variables))
        else:
            self.means_vector = means_vector
            self.cov_matrix = cov_matrix

        self.pm = MultiGauss(list(range(n_variables)), lower_bound, upper_bound)
        self.pm.pm_means = self.means_vector
        self.pm.pm_cov = self.cov_matrix

        self.id = 3

    def sample(self, size) -> np.array:
        return self.pm.sample(size)
