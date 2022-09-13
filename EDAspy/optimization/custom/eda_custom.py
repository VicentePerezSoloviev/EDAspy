#!/usr/bin/env python
# coding: utf-8

from .probabilistic_models import UniBin, UniGauss, GBN, MultiGauss
from .initialization_models import UniformGenInit, MultiGaussGenInit, UniBinGenInit, UniGaussGenInit
from ..eda import EDA


class EDACustom(EDA):

    """
    This class allows the user to define an EDA by custom. This implementation is thought to be extended and extend
    the methods to allow different implementations. Moreover, the probabilistic models and initializations can be
    combined to invent or design a custom EDA.

    The class allows the user to export and load the settings of previous EDA configurations, so this favours the
    implementation of auto-tuning approaches, for example.

    Example:

        This example uses some very well-known benchmarks from CEC14 conference to be solved using
        a custom implementation of EDAs.

        .. code-block:: python

            from EDAspy.optimization.custom import EDACustom, GBN, UniformGenInit
            from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

            n_variables = 10
            my_eda = EDACustom(size_gen=100, max_iter=100, dead_iter=n_variables, n_variables=n_variables, alpha=0.5,
                               elite_factor=0.2, disp=True, pm=4, init=4, bounds=(-50, 50))

            benchmarking = ContinuousBenchmarkingCEC14(n_variables)

            my_gbn = GBN([str(i) for i in range(n_variables)])
            my_init = UniformGenInit(n_variables)

            my_eda.pm = my_gbn
            my_eda.init = my_init

            eda_result = my_eda.minimize(cost_function=benchmarking.cec14_4)


    """

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

        """
        :param size_gen: Population size.
        :param max_iter: Number of maximum iterations.
        :param dead_iter: This is the stopping criteria of the algorithm. After a number of iterations of no
        improvement of the best cost found, the algorithm stops.
        :param n_variables: Number of variables to optimize.
        :param alpha: Percentage [0, 1] of solutions for the truncation of the algorithm in each iteration.
        :param elite_factor: Percentage of population from the last iteration that is appended to the next one,
        in order to implement an elitist approach.
        :param disp: Boolean variable to display of not the final results.
        :param pm: Identifier of probabilistic model to be used in the model: 1 -> univariate Gaussian; 2 ->
        univariate binary; 3 -> multivariate Gaussian; 4 -> Gaussian Bayesian network.
        :param init: Identifier of the initializator to be used in the model: 1 -> univariate Gaussian; 2 ->
        univariate binary; 3 -> multivariate Gaussian; 4 -> uniform.
        :param bounds: tuple with the expected bound of the landscape.
        """

        super().__init__(size_gen, max_iter, dead_iter, n_variables, alpha, elite_factor, disp)

        names_var = list([str(i) for i in range(self.n_variables)])

        # Probabilistic model setting
        if pm == 1:
            self.pm = UniGauss(names_var, lower_bound=bounds[0])
        elif pm == 2:
            self.pm = UniBin(names_var, lower_bound=bounds[0], upper_bound=bounds[1])
        elif pm == 3:
            self.pm = MultiGauss(names_var, lower_bound=bounds[0], upper_bound=bounds[1])
        elif pm == 4:
            self.pm = GBN(names_var)
        else:
            raise ValueError("The probabilistic model is not properly defined.")

        # Initialization model setting
        if init == 1:
            self.init = UniGaussGenInit(self.n_variables, lower_bound=bounds[0])
        elif init == 2:
            self.init = UniBinGenInit(self.n_variables)
        elif init == 3:
            self.init = MultiGaussGenInit(self.n_variables, lower_bound=bounds[0], upper_bound=bounds[1])
        elif init == 4:
            self.init = UniformGenInit(self.n_variables, lower_bound=bounds[0], upper_bound=bounds[1])
        else:
            raise ValueError("The probabilistic model is not properly defined.")

        self.generation = self._initialize_generation()
        self.bounds = bounds

    def export_settings(self) -> dict:
        """
        Export the settings of the EDA.
        :return: dictionary with the configuration.
        :rtype dict
        """

        dic = {
            "size_gen": self.size_gen,
            "max_iter": self.max_iter,
            "dead_iter:": self.dead_iter,
            "n_variables": self.n_variables,
            "pm": self.pm.id,
            "init": self.init.id,
            "alpha": self.alpha,
            "elite_factor": self.elite_factor,
            "disp": self.disp,
            "bounds": self.bounds
        }
        return dic


def read_settings(settings) -> EDACustom:

    """
    This function is implemented to automatic implement the EDA custom by importing the configuration of
    a previous implementation. The function accepts the configuration exported from a previous EDA.

    :param settings: dictionary with the previous configuration.
    :type settings: dict
    :return: EDA custom automatic built.
    :rtype: EDACustom
    """

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
