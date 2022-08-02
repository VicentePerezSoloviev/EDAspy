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
                 bounds: tuple = (-100, 100)):

        super().__init__(n_variables)
        assert len(means_vector) == len(cov_matrix), "Lengths of means vector and covariance matrix must be the same."

        if len(means_vector) == 0:
            self._means_vector = np.random.randint(low=bounds[0], high=bounds[1], size=n_variables)
            self._cov_matrix = np.random.randint(low=bounds[0], high=bounds[1], size=(n_variables, n_variables))
        else:
            self._means_vector = means_vector
            self._cov_matrix = cov_matrix

        self._pm = MultiGauss(list(range(n_variables)), 99999999)
        self._pm.pm_means = self._means_vector
        self._pm.pm_cov = self._cov_matrix

        self.id = 3

    def sample(self, size):
        return self._pm.sample(size)
