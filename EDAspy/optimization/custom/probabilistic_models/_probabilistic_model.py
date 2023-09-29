#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class ProbabilisticModel(ABC):

    def __init__(self, variables: list):
        self.variables = variables
        self.len_variables = len(variables)

        self.id = -1

    @abstractmethod
    def sample(self, size: int):
        raise Exception("Not implemented method")

    @abstractmethod
    def learn(self, dataset: np.array, *args, **kwargs):
        raise Exception("Not implemented method")

    def export_settings(self):
        return self.id

    @abstractmethod
    def print_structure(self) -> list:
        raise Exception("Not implemented method")

