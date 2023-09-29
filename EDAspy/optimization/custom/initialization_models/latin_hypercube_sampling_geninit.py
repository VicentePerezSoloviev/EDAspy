#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import Union, List
from scipy.stats import qmc

from ._generation_init import GenInit


class LatinHypercubeSampling(GenInit):

    """
    Initial generation simulator based on the Latin Hypercube Sampling process.
    """

    def __init__(self, n_variables: int,
                 lower_bound: Union[np.array, List[float], float] = -100,
                 upper_bound: Union[List[float], float] = 100
                 ):
        """
        :param n_variables: Number of variables.
        :param lower_bound: lower bound for the uniform distribution sampling.
        :param upper_bound: lower bound for the uniform distribution sampling.
        :rtype lower_bound: List of lower bounds of size equal to number of variables OR single bound to all dimensions.
        :rtype upper_bound: List of upper bounds of size equal to number of variables OR single bound to all dimensions.
        """
        super().__init__(n_variables)

        if (type(lower_bound) is np.array) or (type(lower_bound) is list):
            assert len(lower_bound) == n_variables, "Number of lower bounds does not match the dimension size."

        if (type(upper_bound) is np.array) or (type(upper_bound) is list):
            assert len(upper_bound) == n_variables, "Number of upper bounds does not match the dimension size."

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.id = 5

    def sample(self, size: int, post_process: bool = False) -> np.array:
        """
        Sample several times the initializer.

        :param size: number of samplings.
        :param post_process: Post processing to ensure diversity between solutions using Lloyd-Max algorithm.
        :return: array with the dataset sampled
        :rtype: np.array
        """
        if post_process:
            sampler = qmc.LatinHypercube(d=self.n_variables)
        else:
            sampler = qmc.LatinHypercube(d=self.n_variables, optimization="lloyd")

        sample = sampler.random(n=size)

        return qmc.scale(sample, self.lower_bound, self.upper_bound)
