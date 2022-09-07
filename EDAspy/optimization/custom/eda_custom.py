#!/usr/bin/env python
# coding: utf-8

import numpy as np
from probabilistic_models import UniBin, UniGauss, GBN, MultiGauss
from initialization_models import UniformGenInit, MultiGaussGenInit, UniBinGenInit, UniGaussGenInit
from ..eda import EDA


class EDACustom(EDA):

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float,
                 elite_factor: float,
                 disp: bool,
                 pm: int,
                 init: int,
                 bounds: tuple
                 ):

        super().__init__(size_gen, max_iter, dead_iter, n_variables, alpha, elite_factor, disp)

        names_var = list(range(self.n_variables))

        # Probabilistic model setting
        if pm == 1:
            self._pm = UniGauss(names_var, lower_bound=bounds[0], upper_bound=bounds[1])
        elif pm == 2:
            self._pm = UniBin(names_var, lower_bound=bounds[0], upper_bound=bounds[1])
        elif pm == 3:
            self._pm = MultiGauss(names_var, lower_bound=bounds[0], upper_bound=bounds[1])
        elif pm == 4:
            self._pm = GBN(names_var)
        else:
            raise ValueError("The probabilistic model is not properly defined.")

        # Initialization model setting
        if init == 1:
            self._init = UniGaussGenInit(self.n_variables, lower_bound=bounds[0], upper_bound=bounds[1])
        elif init == 2:
            self._init = UniBinGenInit(self.n_variables)
        elif init == 3:
            self._init = MultiGaussGenInit(self.n_variables, lower_bound=bounds[0], upper_bound=bounds[1])
        elif init == 4:
            self._init = UniformGenInit(self.n_variables, lower_bound=bounds[0], upper_bound=bounds[1])
        else:
            raise ValueError("The probabilistic model is not properly defined.")

        self.generation = self._initialize_generation()
        self.bounds = bounds

    def _new_generation(self):
        self.generation = self._pm.sample(self.size_gen)

    def _update_pm(self):
        self._pm.learn(self.generation)

    def _initialize_generation(self):
        return self._init.sample(self.size_gen)

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
            "disp": self.disp,
            "bounds": self.bounds
        }
        return dic


def read_settings(settings) -> EDACustom:

    eda = EDACustom(size_gen=settings["size-gen"],
                    max_iter=settings["max_iter"],
                    dead_iter=settings["dead_iter"],
                    n_variables=settings["n_variables"],
                    alpha=settings["alpha"],
                    elite_factor=settings["elite_factor"],
                    disp=settings["disp"],
                    pm=settings["pm"],
                    init=settings["init"],
                    bounds=settings["bounds"]
                    )
    return eda
