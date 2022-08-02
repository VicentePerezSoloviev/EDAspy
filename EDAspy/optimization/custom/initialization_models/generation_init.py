#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod


class GenInit(ABC):

    def __init__(self, n_variables):
        self.n_variables = n_variables

        self._id = -1

    @abstractmethod
    def sample(self, size):
        raise Exception("Not implemented method")

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @n_variables.setter
    def n_variables(self, value: int):
        if value <= 0:
            raise ValueError('The number of variables must be greater than zero.')

        self._n_variables = value

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        assert value > 0, "ID must be greater than zero."
        self._id = value
