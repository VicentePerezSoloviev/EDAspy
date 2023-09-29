#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import Union, List
import pandas as pd

from ._generation_init import GenInit


class CategoricalSampling(GenInit):

    """
    Initial generation simulator based on the Latin Hypercube Sampling process.
    """

    def __init__(self, n_variables: int,
                 possible_values: Union[List, np.array],
                 frequency: Union[List, np.array]
                 ):
        """
        :param n_variables: Number of variables.
        """
        super().__init__(n_variables)

        for i in range(n_variables):
            assert len(possible_values[i]) == len(frequency[i]), "Associated frequency to possible values do not match."
            assert abs(np.sum(frequency[i]) - 1) <= 1e-1, "The frequencies do not sum 1, even considering " \
                                                          "tolerance errors."

        self.possible_values = possible_values
        self.frequency = frequency

        self.id = 6

    def sample(self, size: int) -> np.array:
        """
        Sample several times the initializer.

        :param size: number of samplings.
        :return: array with the dataset sampled
        :rtype: np.array
        """
        data = {}
        for i in range(self.n_variables):
            data[i] = np.random.choice(self.possible_values[i], size=size)

        return pd.DataFrame(data).to_numpy()
