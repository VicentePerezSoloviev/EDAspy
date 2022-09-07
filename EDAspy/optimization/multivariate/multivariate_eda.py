#!/usr/bin/env python
# coding: utf-8

from abc import abstractmethod
import numpy as np
from ..eda import EDA


class MultivariateEda(EDA):

    # TODO: implement getters and setters and set private attributes

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 landscape_bounds: tuple,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True):

        self.landscape_bounds = landscape_bounds
        self.vars = list(range(self.n_variables))
        super().__init__(size_gen, max_iter, dead_iter, n_variables, alpha, elite_factor, disp)

    def _initialize_generation(self):
        return np.random.randint(self.landscape_bounds[0], self.landscape_bounds[1],
                                 (self.size_gen, self.n_variables)).astype(float)

    @abstractmethod
    def _update_pm(self):
        raise Exception("Not implemented method")

    @abstractmethod
    def _new_generation(self):
        raise Exception("Not implemented method")

    @property
    def landscape_bounds(self) -> tuple:
        return self._landscape_bounds

    @landscape_bounds.setter
    def landscape_bounds(self, value):
        if len(value) != 2:
            raise ValueError("The bound are not properly defined")
        self._landscape_bounds = value
