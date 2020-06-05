#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import rpy2.robjects.packages as rp
import numpy as np

from .__BayesianNetwork import learn_structure, calculate_fit
from .__clustering import clustering
from .__matrix import nearestPD, normalizacion, is_invertible

utils = rp.importr('utils')
utils.chooseCRANmirror(ind=1)
bnlearn_package = rp.importr("bnlearn")
dbnRPac = rp.importr("dbnR")


class EDAgbn:

    """Multivariate Estimation of Distribution algorithm. Best individuals of each generation are selected and modelled
    by a Gaussian Bayesian network, from where new individuals are sampled. Some of the variables might be evidences
    (fixed values). The optimizer will find the optimum values of the non-evidences variables for those evidences.
    Also it is possible to control de influence of the historic (from which EDA is initialized) in the selection
    of the best indiviudals.

    :param MAX_ITER: Maximum number of iterations of the algorithm
    :type MAX_ITER: int
    :param DEAD_ITER: Number of iterations with no best cost improvement, before stopping
    :type DEAD_ITER: int
    :param data: data of the historic
    :type data: pandas dataframe
    :param ALPHA: percentage of population to select in the truncation
    :type ALPHA: float [0,1]
    :param BETA: percentage of influence of the individual likelihood in the historic
    :type BETA: float [0,1]
    :param cost_function: cost function to minimize
    :type cost_function: callable function which receives a dictionary as input and returns a numeric value
    :param evidences: name of evidences variables, and fixed values.
    :type evidences: two fold list. A list that contains list of size 2 with name of variable and value [name, value]
    :param black_list: forbidden arcs in the structures
    :type black_list: pandas dataframe with two columns (from, to)
    :param n_clusters: number of clusters in which, the data can be grouped. The cluster is appended in each iteration
    :type n_clusters: int
    :param cluster_vars: list of names of variables to consider in the clustering
    :type cluster_vars: list of strings

    :raises Exception: cost function is not callable

    """

    # initializations
    # structure = -1

    # best_ind_global = -1
    # best_cost_global = 999999999999
    # best_structure = -1
    # history = -1
    # history_cost = []
    # dispersion = []

    def __init__(self, MAX_ITER, DEAD_ITER, data, ALPHA, BETA, cost_function,
                 evidences, black_list, n_clusters, cluster_vars):

        """Constructor method
        """

        self.MAX_ITER = MAX_ITER
        self.DEAD_ITER = DEAD_ITER
        self.data = data
        self.length = int(len(data) * ALPHA)
        self.beta = BETA

        if callable(cost_function):
            self.cost_function = cost_function
        else:
            raise Exception('ERROR setting cost function. Cost function must be callable')

        self.evidences = evidences

        # calculate historic fit
        self.historic_fit = calculate_fit(data, 'hc', black_list)
        self.black_list = black_list
        self.generation = data

        # definition of the variables to optimize
        ev = [row[0] for row in evidences]
        columns = list(data.columns)
        self.variables2optimize = list(set(columns) - set(ev))

        # clustering to soft restrictions
        cluster = clustering(n_clusters, data, evidences, cluster_vars)
        # add cost to clustering selection
        indexes = list(cluster.index)
        dic = {}
        for index in indexes:
            for var in self.variables2optimize:
                dic[var] = float(cluster.loc[index, var])

            cost = self.cost_function(dic)
            cluster.loc[index, 'COSTE'] = cost
        self.dt_aux = cluster.nsmallest(self.length, 'COSTE')  # assign cost to clustering selection

        # initializations
        self.__best_cost_global = 99999999999
        self.__best_structure = -1
        self.__history = pd.DataFrame(columns=data.columns)
        self.__history_cost = []
        self.__dispersion = []
        self.__best_ind_global = -1

        self.structure = -1

    def __initialize_data__(self):

        """Initialize the dataset. Assign a column cost to each individual
        """

        indexes = list(self.generation.index)
        dic = {}
        for index in indexes:
            for var in self.variables2optimize:
                dic[var] = float(self.generation.loc[index, var])

            cost = self.cost_function(dic)
            self.generation.loc[index, 'COSTE'] = cost

    def truncate(self):

        """Select the best individuals of the generation. In this case, not only the cost is considered. Also the
        likelihood of the individual in the initial generation. This influence is controlled by beta parameter
        """

        likelihood_estimation = bnlearn_package.logLik_bn_fit
        names = list(self.generation.columns)
        to_test = list(set(names) - {'COSTE'})

        logs = []
        for index, row in self.generation[to_test].iterrows():
            log = likelihood_estimation(self.historic_fit, pd.DataFrame(row).T)[0]
            logs.append([index, log])

        maximo = max([row[1] for row in logs])
        for i in logs:
            i[1] = maximo - i[1]
            value = float(self.generation.loc[i[0], 'COSTE']) + self.beta * abs(i[1])
            self.generation.loc[i[0], 'trun'] = value

        self.generation = self.generation.nsmallest(self.length, 'trun')
        del self.generation['trun']

    def sampling_multivariate(self, fit):

        """Calculate the parameters mu and sigma to sample from a multivariate normal distribution.

        :param fit: bnfit object from R of the generation (structure and data)
        :type fit: bnfit object from R
        :return: name in order of the parameters returned. mu and sigma parameters of the multivariate normal distribution
        :rtype: list, float, float
        """

        hierarchical_order = bnlearn_package.node_ordering(self.structure)
        mu_calc, sigma_calc = dbnRPac.calc_mu, dbnRPac.calc_sigma

        # mus array -> alphabetic order
        mat_mus = mu_calc(fit)

        # covariance matrix hierarchical order
        cov = list(sigma_calc(fit))
        split = len(mat_mus)
        mat_cov = np.array(cov).reshape(split, split)

        # change order of the columns and rows. Up-left must be variables and down right evidences
        sampled_nodes = [row[0] for row in self.evidences]
        sampled_value = [row[1] for row in self.evidences]
        nodes2sample = list(set(hierarchical_order) - set(sampled_nodes))

        # order evidences and variables to sample in order jerarquico
        nodes2sample_hierarchical_order = []
        nodes_sampled_hierarchical_order = []

        for i in hierarchical_order:
            if i in nodes2sample:
                nodes2sample_hierarchical_order.append(i)
            else:
                nodes_sampled_hierarchical_order.append(i)

        mat_cov_order = nodes2sample_hierarchical_order + nodes_sampled_hierarchical_order  # desired order
        order = []
        for i in mat_cov_order:
            order.append(hierarchical_order.index(i))  # place of each order[] in hierarchical order

        # covariance matrix
        mat_cov[list(range(len(hierarchical_order))), :] = mat_cov[order, :]  # rows swap
        mat_cov[:, list(range(len(hierarchical_order)))] = mat_cov[:, order]  # columns swap
        mat_cov_evid = np.array([row[len(nodes2sample):] for row in mat_cov[len(nodes2sample):]])

        from numpy.linalg import inv, pinv
        if is_invertible(mat_cov):
            mat_cov_inv = inv(mat_cov)
        else:
            mat_cov_inv = pinv(mat_cov, hermitian=True)

        mat_cov_inv_data = np.array([row[:len(nodes2sample)] for row in mat_cov_inv[:len(nodes2sample)]])
        sum_12 = [row[len(nodes2sample):] for row in mat_cov[:len(nodes2sample)]]

        orden_mat_mus = sorted(hierarchical_order)  # order hierarchical en order alphabetic

        mus = []  # mus en order hierarchical
        for i in hierarchical_order:  # append in mus, mu calc of the position where is i
            mus.append(mat_mus[orden_mat_mus.index(i)])

        values = []  # sampled values in hierarchical order
        order_values = []
        for i in hierarchical_order:
            if i not in nodes2sample:
                order_values.append(i)
                values.append(sampled_value[sampled_nodes.index(i)])

        # mus en order hierarchical
        mat_mus_data = []
        mat_mus_evid = []

        for i in hierarchical_order:
            if i in nodes2sample:
                mat_mus_data.append(mus[hierarchical_order.index(i)])
            else:
                mat_mus_evid.append(mus[hierarchical_order.index(i)])

        # FORMULAS MURPHY
        aux = np.subtract(np.array(values), np.array(mat_mus_evid))
        aux1 = np.matmul(sum_12, pinv(mat_cov_evid))
        aux2 = np.matmul(aux1, aux)

        mu_cond = np.add(mat_mus_data, np.array(aux2))

        if is_invertible(mat_cov):
            mat_cov_inv_data_inv = inv(mat_cov_inv_data)
        else:
            mat_cov_inv_data_inv = pinv(mat_cov_inv_data, hermitian=True)

        return nodes2sample_hierarchical_order, mu_cond, mat_cov_inv_data_inv  # , densities

    def new_generation(self):

        """Build a new generation from the parameters calculated and update the generation to the new group of individuals

        :return: mean and sigma of the individuals costs of the generation
        :rtype: float, float
        """

        valid_individuals = 0
        hierarchical_order = bnlearn_package.node_ordering(self.structure)

        bn_fit = bnlearn_package.bn_fit
        fit = bn_fit(self.structure, self.generation)

        nodes, mu, sigma = self.sampling_multivariate(fit)

        # precision errors solved
        sigma = nearestPD(np.array(sigma))
        gen = pd.DataFrame(columns=hierarchical_order)
        media = []
        counter = 0

        while valid_individuals < len(self.data):
            sample = np.random.multivariate_normal(mu, sigma, 1)[0]
            # normalization
            cost_position = nodes.index('COSTE')
            values = [item for item in sample if list(sample).index(item) != cost_position]
            ind_ = normalizacion(values)
            ind = ind_[:cost_position] + [sample[cost_position]] + ind_[cost_position:]

            # append the evidences
            individual = self.evidences[:]
            for j in range(len(ind)):
                individual.append([nodes[j], ind[j]])

            # avoid solutions with the parameters under zero
            flag = True
            for i in individual:
                for j in self.variables2optimize:
                    if i[0] == j:
                        if i[1] < 0:
                            flag = False

            # if restrictions are correct
            if flag:
                dic = {}
                for i in individual:
                    aux = {i[0]: i[1]}
                    dic.update(aux)

                cost = self.cost_function(dic)
                dic.update({'COSTE': cost})
                gen = gen.append(dic, ignore_index=True)

                valid_individuals = valid_individuals + 1
                media.append(cost)
                counter = 0

            else:
                counter = counter + 1
                if counter == len(self.data) * 10:
                    break

        self.generation = gen.copy()

        from scipy.stats import norm
        mu, sigma = norm.fit(media)
        return mu, sigma

    def soft_restrictions(self, NOISE):

        """Add Gaussian noise to the evidence variables

        :param NOISE: std of the normal distribution from where noise is sampled
        :type NOISE: float
        """

        number_samples = len(self.generation)

        # generate noise from a normal distribution
        for i in self.evidences:
            s = np.random.normal(i[1], NOISE, number_samples)
            self.generation[i[0]] = s

    def __choose_best__(self):

        """Select the best individual of the generation
        :return: cost of the individual, and the individual
        :rtype: float, pandas dataframe
        """

        minimum = self.generation['COSTE'].min()
        best_ind_local = self.generation[self.generation['COSTE'] == minimum]
        return minimum, best_ind_local

    def run(self, output=True):
        """Running method
        :param output: if desired to print the output of each individual. True to print output
        :type output: boolean
        :return:the class is returned, in order to explore all the parameters
        :rtype: self python class
        """

        self.__initialize_data__()
        dead_iter = 0

        for ITER in range(self.MAX_ITER):
            self.truncate()

            # soft the values of the evidences variables
            if ITER > 0:
                self.soft_restrictions(0.01)
                self.generation = self.generation.append(self.dt_aux, ignore_index=True)
            else:
                # first iteration
                self.structure = learn_structure(self.generation, 'hc', self.black_list)

            # print_structure(self.structure, self.variables2optimize, [row[0] for row in self.evidences])
            mu, sigma = self.new_generation()
            self.__dispersion.append([mu, sigma])  # info about dispersion among individuals in each iteration
            self.structure = learn_structure(self.generation, 'hc', self.black_list)

            # if there have not been found all individuals
            if len(self.generation) < len(self.data) / 2:
                return self

            # set local and update global best
            best_cost_local, best_ind_local = self.__choose_best__()
            self.__history.append(best_ind_local, ignore_index=True)
            self.__history_cost.append(best_cost_local)

            # update the global cost, structure and best individual, if needed
            if best_cost_local < self.__best_cost_global:
                self.__best_cost_global = best_cost_local
                self.__best_ind_global = best_ind_local
                self.__best_structure = self.structure
                dead_iter = 0
            else:
                dead_iter = dead_iter + 1
                if dead_iter == self.DEAD_ITER:
                    return self

            if output:
                print('ITER: ', ITER, 'dead: ', dead_iter,
                      'bestLocal: ', best_cost_local, 'bestGlobal: ', self.best_cost_global)

        return self

    """
    Getters of interesting attributes
    """

    @property
    def best_cost_global(self):
        """
        :return: best cost found ever at the end of the execution
        :rtype: float
        """
        return self.__best_cost_global

    @property
    def best_ind_global(self):
        """
        :return: best individual found ever at the end of the execution
        :rtype: pandas dataframe slice
        """
        return self.__best_ind_global

    @property
    def best_structure(self):
        """
        :return: best generation structure found ever at the end of the execution
        :rtype: bnlearn structure
        """
        return self.__best_structure

    @property
    def history(self):
        """
        :return: best individuals from all generations
        :rtype: pandas dataframe
        """
        return self.__history

    @property
    def history_cost(self):
        """
        :return: list of best costs found along the execution
        :rtype: list
        """
        return self.__history_cost

    @property
    def dispersion(self):
        """
        :return: list of double tuples with mean and variance of each generation
        :rtype: list
        """
        return self.__dispersion
