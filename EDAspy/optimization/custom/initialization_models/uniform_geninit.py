#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .generation_init import GenInit


class UniformGenInit(GenInit):

    def __init__(self,
                 n_variables: int,
                 bounds: tuple = (-100, 100)):

        super().__init__(n_variables)

        self.bounds = bounds

        self.id = 4

    def sample(self, size):
        return np.random.randint(self.bounds[0], self.bounds[1], (self.n_variables, size)).astype(float)

    @property
    def bounds(self) -> tuple:
        return self._bounds

    @bounds.setter
    def bounds(self, value: tuple):
        assert value[0] < value[1], "Minimum should be lower than the maximum in the defined tuple."
        self._bounds = value
