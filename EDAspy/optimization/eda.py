#!/usr/bin/env python
# coding: utf-8

import numpy as np
from abc import ABC


class EDA(ABC):

    """
    Abstract class which defines the general performance of the algorithms. The baseline of the EDA
    approach is defined in this object. The specific configurations is defined in the class of each
    specific algorithm.
    """

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

        self.pm = None
        self.init = None
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

    # run the class to find the optimum
    def minimize(self, cost_function: callable, output_runtime: bool = True):
        r"""
        Args:
            cost_function: Cost function to be optimized and accepts an array as argument.
            output_runtime: True if information during runtime is desired.

        :return: EdaResult object
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


class EdaResult:

    """
    Object used to encapsulate the result and information of the EDA during the execution
    """

    def __init__(self,
                 best_ind: np.array,
                 best_cost: float,
                 n_fev: int,
                 history: list,
                 settings: dict):

        """

        :param best_ind: Best result found in the execution.
        :param best_cost: Cost of the best result found.
        :param n_fev: Number of cost function evaluations.
        :param history: Best result found in each iteration of the algorithm.
        :param settings: Configuration of the parameters of the EDA.
        """

        self.best_ind = best_ind
        self.best_cost = best_cost
        self.n_fev = n_fev
        self.history = history
        self.settings = settings
