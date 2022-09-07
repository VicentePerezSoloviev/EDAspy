#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
import numpy as np
from . import UniBin, UniGauss, GBN, MultiGauss


class ProbabilisticModel(ABC):

    def __init__(self, variables: list):
        self.variables = variables
        self.len_variables = len(variables)

        self._id = -1

    @abstractmethod
    def sample(self, size: int):
        raise Exception("Not implemented method")

    @abstractmethod
    def learn(self, dataset: np.array):
        raise Exception("Not implemented method")

    @abstractmethod
    def export_settings(self):
        raise Exception("Not implemented method")

    @property
    def variables(self) -> list:
        """Returns the names of the variables of the probabilistic model"""
        return self._variables

    @variables.setter
    def variables(self, variables: list):
        """
        Sets the names of the variables.
        Args:
            variables: list of variables
        Raises:
            ValueError: If list is empty
        """
        if len(variables) == 0:
            raise ValueError("The number of variables in the probabilistic model must be greater than 0.")
        self._variables = variables

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        assert value > 0, "ID must be greater than zero."
        self._id = value
