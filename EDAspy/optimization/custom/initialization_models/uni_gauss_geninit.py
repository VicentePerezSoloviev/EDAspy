#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ..probabilistic_models import UniGauss
from .generation_init import GenInit


class UniGaussGenInit(GenInit):

    def __init__(self,
                 n_variables: int,
                 means_vector: np.array = np.empty(0),
                 stds_vector: np.array = np.empty(0),
                 lower_bound: int = -100,
                 upper_bound: int = 100):

        super().__init__(n_variables)

        assert len(means_vector) == len(stds_vector), "Lengths of means and stds vector must be the same."

        if len(means_vector) == 0:
            self._means_vector = np.random.randint(low=lower_bound, high=upper_bound, size=n_variables)
            self._stds_vector = np.random.randint(low=lower_bound, high=upper_bound, size=n_variables)
        else:
            self._means_vector = means_vector
            self._stds_vector = stds_vector

        self._pm = UniGauss(list(range(self.n_variables)), lower_bound, upper_bound)
        self._pm.pm = np.array([means_vector, stds_vector])

        self.id = 1

    def sample(self, size):
        return self._pm.sample(size=size)
