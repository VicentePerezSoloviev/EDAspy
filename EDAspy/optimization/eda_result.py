#!/usr/bin/env python
# coding: utf-8

import numpy as np


class EdaResult:

    """
    Object used to encapsulate the result and information of the EDA during the execution
    """

    def __init__(self,
                 best_ind: np.array,
                 best_cost: float,
                 n_fev: int,
                 history: list,
                 settings: dict,
                 cpu_time: float):

        """

        :param best_ind: Best result found in the execution.
        :param best_cost: Cost of the best result found.
        :param n_fev: Number of cost function evaluations.
        :param history: Best result found in each iteration of the algorithm.
        :param settings: Configuration of the parameters of the EDA.
        :param cpu_time: CPU time invested in the optimization.
        """

        self.best_ind = best_ind
        self.best_cost = best_cost
        self.n_fev = n_fev
        self.history = history
        self.settings = settings
        self.cpu_time = cpu_time

    def __str__(self):
        string = "\tNFVALS = " + str(self.n_fev) + " F = " + str(self.best_cost) + "CPU time (s) = " + \
                 str(self.cpu_time) + "\n\tX = " + str(self.best_ind) + "\n\tSettings: " + str(self.settings) + \
                 "\n\tHistory best cost per iteration: " + str(self.history)
        return string
