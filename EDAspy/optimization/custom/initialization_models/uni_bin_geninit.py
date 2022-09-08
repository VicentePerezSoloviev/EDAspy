#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ..probabilistic_models import UniBin
from .generation_init import GenInit


class UniBinGenInit(GenInit):

    def __init__(self, n_variables, means_vector: np.array = np.empty(0)):
        super().__init__(n_variables)

        if len(means_vector) == 0:
            self.means_vector = np.array([0.5] * self.n_variables)
        else:
            self.means_vector = means_vector

        self.pm = UniBin(list(range(self.n_variables)), lower_bound=0, upper_bound=0)  # dismiss bounds
        self.pm.pm = means_vector

        self.id = 2

    def sample(self, size) -> np.array:
        return self.pm.sample(size)
