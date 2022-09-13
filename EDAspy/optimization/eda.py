#!/usr/bin/env python
# coding: utf-8

import numpy as np
from abc import ABC
from .eda_result import EdaResult
from .custom.probabilistic_models import ProbabilisticModel
from .custom.initialization_models import GenInit


class EDA(ABC):

    """
    Abstract class which defines the general performance of the algorithms. The baseline of the EDA
    approach is defined in this object. The specific configurations is defined in the class of each
    specific algorithm.
    """

    _pm = None
    _init = None

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True):

        self.disp = disp
        self.size_gen = size_gen
        self.max_iter = max_iter
        self.alpha = alpha
        self.n_variables = n_variables
        self.truncation_length = int(size_gen * alpha)
        self.elite_factor = elite_factor
        self.elite_length = int(size_gen * elite_factor)

        assert dead_iter <= self.max_iter, 'dead_iter must be lower than max_iter'
        self.dead_iter = dead_iter

        self.best_mae_global = 999999999999
        self.best_ind_global = -1
        self.evaluations = np.array(0)

        self.generation = None

    def _new_generation(self):
        self.generation = np.concatenate([self.pm.sample(size=self.size_gen), self.elite_temp])

    def _initialize_generation(self) -> np.array:
        return self.init.sample(size=self.size_gen)

    def _truncation(self):
        """
        Selection of the best individuals of the actual generation.
        """
        ordering = self.evaluations.argsort()
        best_indices_truc = ordering[: self.truncation_length]
        best_indices_elit = ordering[: self.elite_length]
        self.elite_temp = self.generation[best_indices_elit, :]
        self.generation = self.generation[best_indices_truc, :]
        self.evaluations = np.take(self.evaluations, best_indices_truc)

    # check each individual of the generation
    def _check_generation(self, objective_function):
        """
        Check the cost of each individual in the cost function implemented by the user, and updates the
        generation DataFrame.
        """
        self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)

    def _update_pm(self):
        """
        Learn the probabilistic model from the best individuals of previous generation.
        """
        self.pm.learn(dataset=self.generation)

    def export_settings(self) -> dict:
        """
        Export the configuration of the algorithm to an object to be loaded in other execution.
        :return: dict
        """
        return {
            "size_gen": self.size_gen,
            "max_iter": self.max_iter,
            "dead_iter": self.dead_iter,
            "n_variables": self.n_variables,
            "alpha": self.alpha,
            "elite_factor": self.elite_factor,
            "disp": self.disp
        }

    def minimize(self, cost_function: callable, output_runtime: bool = True):
        r"""
        :param cost_function: cost function to be optimized and accepts an array as argument.
        :param output_runtime: true if information during runtime is desired.
        :return: EdaResult object
        :rtype: EdaResult
        """

        history = []
        not_better = 0

        self.generation = self._initialize_generation()

        for _ in range(self.max_iter):
            self._check_generation(cost_function)
            self._truncation()
            self._update_pm()

            best_mae_local = min(self.evaluations)

            history.append(best_mae_local)
            best_ind_local = np.where(self.evaluations == best_mae_local)[0][0]
            best_ind_local = self.generation[best_ind_local]

            # update the best values ever
            if best_mae_local < self.best_mae_global:
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0

            else:
                not_better += 1
                if not_better == self.dead_iter:
                    break

            self._new_generation()

            if output_runtime:
                print('IT: ', _, '\tBest cost: ', self.best_mae_global)

        if self.disp:
            print("\tNFVALS = " + str(len(history) * self.size_gen) + " F = " + str(self.best_mae_global))
            print("\tX = " + str(self.best_ind_global))

        eda_result = EdaResult(self.best_ind_global, self.best_mae_global, len(history) * self.size_gen,
                               history, self.export_settings())

        return eda_result

    @property
    def pm(self):
        return self._pm

    @pm.setter
    def pm(self, value):
        if isinstance(value, ProbabilisticModel):
            self._pm = value
        else:
            raise ValueError('The object you try to set as a probabilistic model does not extend the '
                             'class ProbabilisticModel provided by EDAspy')

        if len(value.variables) != self.n_variables:
            raise Exception('The number of variables of the probabilistic model is not equal to the number of '
                            'variables of the EDA')

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, value):
        if isinstance(value, GenInit):
            self._init = value
        else:
            raise ValueError('The object you try to set as an initializator does not extend the '
                             'class GenInit provided by EDAspy')

        if value.n_variables != self.n_variables:
            raise Exception('The number of variables of the initializator is not equal to the number of '
                            'variables of the EDA')
