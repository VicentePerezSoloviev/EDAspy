#!/usr/bin/env python
# coding: utf-8

import numpy as np
from probabilistic_models.probabilistic_model import ProbabilisticModel
from initialization_models.generation_init import GenInit
from ..eda import EDA


class EDACustom(EDA):

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 pm: ProbabilisticModel,
                 initialization: GenInit,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True
                 ):

        super().__init__(size_gen, max_iter, dead_iter, n_variables, alpha, elite_factor, disp)

        self._pm = pm
        self._init = initialization

    def read_settings(self, settings):
        super().__init__(settings["size_gen"],
                         settings["max_iter"],
                         settings["dead_iter"],
                         settings["n_variables"],
                         settings["alpha"],
                         settings["elite_factor"],
                         settings["disp"])

        self._pm =
        self._init = initialization

    def export_settings(self):
        dic = {
            "size_gen": self.size_gen,
            "max_iter": self.max_iter,
            "dead_iter:": self.dead_iter,
            "n_variables": self.n_variables,
            "pm": self._pm.id,
            "init": self._init.id,
            "alpha": self.alpha,
            "elite_factor": self.elite_factor,
            "disp": self.disp
        }
        return dic

    def minimize(self, cost_function: callable, output_runtime: bool = True):
        r"""
        Args:
            cost_function: Cost function to be optimized and accepts an array as argument.
            output_runtime: True if information during runtime is desired.
        """

        history = []
        not_better = 0

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

        return self.best_ind_global, self.best_mae_global, len(history) * self.size_gen