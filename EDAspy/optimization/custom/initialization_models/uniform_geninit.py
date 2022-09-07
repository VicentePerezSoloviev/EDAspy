#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .generation_init import GenInit


class UniformGenInit(GenInit):

    def __init__(self,
                 n_variables: int,
                 lower_bound: float = -100,
                 upper_bound: float = 100
                 ):

        super().__init__(n_variables)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.id = 4

    def sample(self, size):
        return np.random.randint(self.lower_bound, self.upper_bound, (self.n_variables, size)).astype(float)
