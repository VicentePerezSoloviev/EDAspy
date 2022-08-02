#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pybnesian import GaussianNetwork
from .probabilistic_model import ProbabilisticModel
import pandas as pd


class GBN(ProbabilisticModel):

    def __init__(self, variables: list):
        super().__init__(variables)

        self._pm = GaussianNetwork(variables)

        self.id = 4

    def learn(self, dataset: np.array):
        self._pm.fit(pd.DataFrame(dataset))

    def print_structure(self):
        return self._pm.arcs()

    def sample(self, size: int):
        dataset = self._pm.sample(size).to_pandas()
        dataset = dataset[self.variables].to_numpy()
        return dataset
