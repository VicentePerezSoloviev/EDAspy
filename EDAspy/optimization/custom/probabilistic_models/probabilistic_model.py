#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
import numpy as np


class ProbabilisticModel(ABC):

    def __init__(self, variables: list):
        self.variables = variables
        self.len_variables = len(variables)

        self.id = -1

    @abstractmethod
    def sample(self, size: int):
        raise Exception("Not implemented method")

    @abstractmethod
    def learn(self, dataset: np.array):
        raise Exception("Not implemented method")

    @abstractmethod
    def export_settings(self):
        return self.id

