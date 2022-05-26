#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
import numpy as np


class MultivariateEda(ABC):

    best_global_ind = np.array(0)

    history = []
    evaluations = []

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 landscape_bounds: tuple,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True):
        r"""
        Args:
            size_gen: Population size. Number of individuals in each generation.
            max_iter: Maximum number of iterations during runtime.
            dead_iter: Stopping criteria. Number of iterations with no improvement after which, the algorithm finish.
            n_variables: Number of variables to be optimized.
            landscape_bounds: Landscape bounds. Limits in the search space.
            alpha: Percentage of population selected to update the probabilistic model.
            elite_factor: Percentage of previous population selected to add to new generation (elite approach).
            disp: Set to True to print convergence messages.
        """

        self.max_it = max_iter
        self.size_gen = size_gen

        assert dead_iter <= max_iter, "dead_it must be lower than max_it"
        self.dead_iter = dead_iter

        self.trunc_size = int(size_gen * alpha)
        self.n_variables = n_variables
        self.elite_factor = elite_factor
        self.disp = disp
        self.landscape_bounds = landscape_bounds
        self.vars = [str(num) for num in range(n_variables)]

        self._initialization()

    def _initialization(self):
        self.generation = np.random.randint(self.landscape_bounds[0], self.landscape_bounds[1],
                                            (self.size_gen, self.n_variables)).astype(float)

    def _evaluation(self, objective_function):
        self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)

    def _truncation(self):
        best_indices = self.evaluations.argsort()[: self.trunc_size]
        self.generation = self.generation[best_indices, :]
        self.evaluations = np.take(self.evaluations, best_indices)

    @abstractmethod
    def _update_pm(self):
        raise Exception("Not implemented method")

    @abstractmethod
    def _new_generation(self):
        raise Exception("Not implemented method")

    def minimize(self, cost_function: callable, output_runtime: bool = True):
        r"""
        Args:
            cost_function: Cost function to be optimized and accepts an array as argument.
            output_runtime: True if information during runtime is desired.
        """
        no_improvement = 0

        for _ in range(self.max_it):
            if no_improvement == self.dead_iter:
                break

            self._evaluation(cost_function)
            self._truncation()
            self._update_pm()

            best_local_cost = self.evaluations[0]
            if best_local_cost < self.best_global_cost:
                self.best_global_cost = best_local_cost
                self.best_global_ind = self.generation[0]
                no_improvement = 0
            else:
                no_improvement += 1

            if output_runtime:
                print('IT: ', _, '\tBest cost: ', self.best_global_cost)

            self._new_generation()
            self.history.append(best_local_cost)

        if self.disp:
            print("\tNFVALS = " + str(len(self.history) * self.size_gen) + " F = " + str(self.best_global_cost))
            print("\tX = " + str(self.best_global_ind))

        return self.best_global_ind, self.best_global_cost, len(self.history) * self.size_gen
