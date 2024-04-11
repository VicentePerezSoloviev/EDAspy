#!/usr/bin/env python
# coding: utf-8

import numpy as np
from abc import ABC
from .eda_result import EdaResult
from .custom.probabilistic_models import ProbabilisticModel
from .custom.initialization_models import GenInit
from .utils import _parallel_apply_along_axis
from time import process_time


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
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: np.array = None,
                 *args, **kwargs):

        self.disp = disp
        self.size_gen = size_gen
        self.max_iter = max_iter
        self.alpha = alpha
        self.n_variables = n_variables
        self.truncation_length = int(size_gen * alpha)
        self.elite_factor = elite_factor
        self.elite_length = int(size_gen * elite_factor)
        self.parallelize = parallelize

        assert dead_iter <= self.max_iter, 'dead_iter must be lower than max_iter'
        self.dead_iter = dead_iter

        self.best_mae_global = 999999999999
        self.best_ind_global = np.array([0]*self.n_variables)

        self.evaluations = np.array(0)
        self.evaluations_elite = np.array(0)

        self.generation = None
        self.elite_temp = None

        if parallelize:
            self._check_generation = self._check_generation_parallel
        else:
            self._check_generation = self._check_generation_no_parallel

        # allow initialize EDA with data
        if init_data is not None:
            assert init_data.shape[1] == n_variables, 'The inserted data shape and the number of variables do not match'
            # assert init_data.shape[0] == size_gen, 'The inserted data shape and the generation size do not match'

            self.init_data = init_data
            self._initialize_generation = self._initialize_generation_with_data
        else:
            self._initialize_generation = self._initialize_generation_with_init

    def _new_generation(self):
        # self.generation = np.concatenate([self.pm.sample(size=self.size_gen), self.elite_temp])
        self.generation = self.pm.sample(size=self.size_gen)

    def _initialize_generation_with_data(self) -> np.array:
        return self.init_data

    def _initialize_generation_with_init(self) -> np.array:
        return self.init.sample(size=self.size_gen)

    def _initialize_generation(self) -> np.array:
        raise Exception('Not implemented function')

    def _truncation(self):
        """
        Selection of the best individuals of the actual generation.
        """
        # first add the elite selection to be considered
        self.generation = np.concatenate([self.generation, self.elite_temp])
        self.evaluations = np.append(self.evaluations, self.evaluations_elite)

        # now we truncate
        ordering = self.evaluations.argsort()
        best_indices_truc = ordering[: self.truncation_length]
        best_indices_elit = ordering[: self.elite_length]
        self.elite_temp = self.generation[best_indices_elit, :]
        self.generation = self.generation[best_indices_truc, :]
        self.evaluations_elite = np.take(self.evaluations, best_indices_elit)
        self.evaluations = np.take(self.evaluations, best_indices_truc)

    # check each individual of the generation
    def _check_generation(self, objective_function: callable):
        """
        Check the cost of each individual in the cost function implemented by the user, and updates the
        generation DataFrame.
        """
        raise Exception('Not implemented function')

    def _check_generation_parallel(self, objective_function: callable):
        self.evaluations = _parallel_apply_along_axis(objective_function, 1, self.generation)

    def _check_generation_no_parallel(self, objective_function: callable):
        self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)

    def _update_pm(self):
        """
        Learn the probabilistic model from the best individuals of previous generation.
        """
        self.pm.learn(dataset=self.generation)

    def export_settings(self) -> dict:
        """
        Export the configuration of the algorithm to an object to be loaded in other execution.

        :return: configuration dictionary.
        :rtype: dict
        """
        return {
            "size_gen": self.size_gen,
            "max_iter": self.max_iter,
            "dead_iter": self.dead_iter,
            "n_variables": self.n_variables,
            "alpha": self.alpha,
            "elite_factor": self.elite_factor,
            "disp": self.disp,
            "parallelize": self.parallelize
        }

    def minimize(self, cost_function: callable, output_runtime: bool = True, sanitize: callable = None, *args, **kwargs) -> EdaResult:
        """
        Minimize function to execute the EDA optimization. By default, the optimizer is designed to minimize a cost
        function; if maximization is desired, just add a minus sign to your cost function.

        :param cost_function: cost function to be optimized and accepts an array as argument.
        :param output_runtime: true if information during runtime is desired.
        :param sanitize: run before updating model, can be used to adjust the representation of the generation
        :return: EdaResult object with results and information.
        :rtype: EdaResult
        """

        history = []
        not_better = 0

        t1 = process_time()
        self.generation = self._initialize_generation()
        self._check_generation(cost_function)

        # select just one item to be the elite selection if first iteration
        self.elite_temp = np.array([self.generation[0, :]])
        self.evaluations_elite = np.array([self.evaluations.item(0)])

        best_mae_local = best_mae_global = min(self.evaluations)
        history.append(best_mae_local)
        best_ind_local = best_ind_global = np.where(self.evaluations == best_mae_local)[0][0]

        for _ in range(self.max_iter - 1):
            self._truncation()
            if sanitize is not None:
                sanitize(self.generation)
            self._update_pm()

            self._new_generation()
            self._check_generation(cost_function)

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

            if output_runtime:
                print('IT: ', _, '\tBest cost: ', self.best_mae_global)

        if self.disp:
            print("\tNFEVALS = " + str(len(history) * self.size_gen) + " F = " + str(self.best_mae_global))
            print("\tX = " + str(self.best_ind_global))

        t2 = process_time()
        eda_result = EdaResult(self.best_ind_global, self.best_mae_global, len(history) * self.size_gen,
                               history, self.export_settings(), t2-t1)

        return eda_result

    @property
    def pm(self) -> ProbabilisticModel:
        """
        Returns the probabilistic model used in the EDA implementation.

        :return: probabilistic model.
        :rtype: ProbabilisticModel
        """
        return self._pm

    @pm.setter
    def pm(self, value):
        if isinstance(value, ProbabilisticModel):
            self._pm = value
        else:
            raise ValueError('The object you try to set as a probabilistic model does not extend the '
                             'class ProbabilisticModel provided by EDAspy.')

        if len(value.variables) != self.n_variables:
            raise Exception('The number of variables of the probabilistic model is not equal to the number of '
                            'variables of the EDA.')

    @property
    def init(self) -> GenInit:
        """
        Returns the initializer used in the EDA implementation.

        :return: initializer.
        :rtype: GenInit
        """
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
