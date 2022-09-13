#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod

import numpy as np


class GenInit(ABC):

    def __init__(self, n_variables):
        self.n_variables = n_variables

        self.id = -1

    @abstractmethod
    def sample(self, size) -> np.array:
        raise Exception("Not implemented method")
