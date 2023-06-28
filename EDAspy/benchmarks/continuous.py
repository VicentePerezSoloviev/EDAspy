#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from typing import Union


def _mod_sch_f(z_i, dim):
    if abs(z_i) <= 500:
        return z_i * np.sin(abs(z_i)**0.5)
    elif z_i > 500:
        return (500 - (z_i % 500)) * np.sin(np.sqrt(abs(500 - (z_i % 500)))) - (((z_i - 500)**2) / 10000*dim)
    elif z_i < -500:
        return ((z_i % 500) - 500) * np.sin(np.sqrt(abs((abs(z_i) % 500) - 500))) - (((z_i + 500)**2) / 10000*dim)


def _scaffer_f6_function(x, y):
    return 0.5 + (((np.sin(np.sqrt(x**2 + y**2))**2) - 0.5)/(1 + 0.001*(x**2 + y**2))**2)


class ContinuousBenchmarkingCEC14:

    def __init__(self,
                 dim: int):

        """
        :param dim: number of variables to be evaluated in the cost function
        """

        assert dim in [1, 10, 20, 30, 50, 100], 'The benchmarking set is not implemented for such dimension. Try one ' \
                                                'of 1, 10, 20, 30, 50, 100.'

        self.d = dim
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

    def _load_shift(self, number):
        text_file = open(os.path.join(self.package_directory, 'input_data', "shift_data_" + str(number) + ".txt"), "r")
        shifts = text_file.read().split()
        text_file.close()
        return np.array(shifts[:self.d]).astype(float)

    def _load_rot(self, number):
        matrix_file = open(os.path.join(self.package_directory, 'input_data', "M_" + str(number) +
                                        "_D" + str(self.d) + ".txt"), "r")
        matrix_file_read = matrix_file.read().split()
        matrix_file.close()
        matrix_file_read = np.array(matrix_file_read).astype(float)
        matrix_file_read = np.reshape(matrix_file_read, (self.d, self.d))
        return matrix_file_read

    def bent_cigar_function(self, x: Union[np.array, list]) -> float:
        """
        Bent Cigar function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        result = x[0]**2
        for i in range(1, self.d):
            result += x[i]**2
        result += (result*np.power(10, 6))
        return result

    def discuss_function(self, x: Union[np.array, list]) -> float:
        """
        Discuss function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        result = np.power(10, 6)*(x[0]**2)
        for i in range(1, self.d):
            result += x[i]**2
        return result

    def rosenbrock_function(self, x: Union[np.array, list]) -> float:
        """
        Rosenbrock’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        result = 0
        for i in range(self.d-1):
            result += (100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2)
        return result

    def ackley_function(self, x: Union[np.array, list]) -> float:
        """
        Ackley’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum1 = sum([x[i]**2 for i in range(self.d)])
        sum2 = sum([np.cos(x[i]*2*np.pi) for i in range(self.d)])
        return -20*np.exp(-0.2*np.sqrt(sum1/self.d)) - np.exp(sum2/self.d) + 20 + np.e

    def weierstrass_function(self, x: Union[np.array, list]) -> float:
        """
        Weierstrass Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        a = 0.5
        b = 3
        k_max = 20
        f = 0
        for i in range(self.d):
            aux_f1 = 0
            aux_f2 = 0
            for k in range(k_max):
                aux_f1 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))
            for k in range(k_max):
                aux_f2 += np.power(a, k)*np.cos(2*np.pi*np.power(b, k)*0.5)

            f += aux_f1 - self.d * aux_f2

        return f

    def griewank_function(self, x: Union[np.array, list]) -> float:
        """
        Griewank’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        f = 0
        for i in range(self.d):
            f += x[i]**2/4000
        aux_ = 0
        for i in range(1, self.d):
            aux_ *= np.cos(x[i-1]/np.sqrt(i)) + 1
        return f - aux_

    def rastrigins_function(self, x: Union[np.array, list]) -> float:
        """
        Rastrigin’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        result = 0
        for i in range(len(x)):
            result += (x[i]**2 - 10*np.cos(2*np.pi*x[i]) + 10)
        return result

    def high_conditioned_elliptic_function(self, x: Union[np.array, list]) -> float:
        result = 0
        for i in range(1, len(x) + 1):
            result += ((10**6)**((i-1)/(len(x)-1)))*x[i-1]**2
        return result

    def mod_schwefels_function(self, x: Union[np.array, list]) -> float:
        """
        Modified Schwefel’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        result = 418.9829 * len(x)
        for i in range(1, len(x) + 1):
            z_i = x[i-1] + 4.209687462275036e+002
            result = result - _mod_sch_f(z_i, len(x))
        return result

    def katsuura_function(self, x: Union[np.array, list]) -> float:
        """
        Katsuura Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        aux = 1
        for i in range(1, self.d + 1):
            aux2 = 0
            for j in range(1, 32):
                aux2 += abs((x[i-1]*(2**j)) - round(x[i-1]*(2**j)))/(2**j)
            aux *= (1 + i*aux2)**(10/(self.d**1.2))
        aux = (aux * (10/(self.d**2))) - (10/self.d**2)
        return aux

    def happycat_function(self, x: Union[np.array, list]) -> float:
        """
        HappyCat Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        a = sum([x[i-1]**2 for i in range(1, self.d + 1)])
        result = abs(a - self.d)**0.25
        result += (0.5 * a + sum([x[i-1] for i in range(1, self.d + 1)]))/self.d + 0.5
        return result

    def hgbat_function(self, x: Union[np.array, list]) -> float:
        """
        HGBat Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        a = sum([x[i-1]**2 for i in range(1, self.d + 1)])
        b = sum([x[i-1] for i in range(1, self.d + 1)])
        result = (abs(a**2 - b**2)**0.5) + (0.5*a + b)/self.d + 0.5
        return result

    def expanded_scaffer_f6_function(self, x: Union[np.array, list]):
        """
        Expanded Scaffer’s F6 Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        result = sum([_scaffer_f6_function(x[i - 1], x[i]) for i in range(1, self.d)])
        result += _scaffer_f6_function(x[self.d - 1], x[0])
        return result

    def cec14_1(self, x: Union[np.array, list]) -> float:
        """
        Rotated High Conditioned Elliptic Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = np.array(x) - self._load_shift(1)
        mat = self._load_rot(1)
        f1 = self.high_conditioned_elliptic_function(np.dot(mat, sum_vectors)) + 100
        return f1

    def cec14_2(self, x: Union[np.array, list]) -> float:
        """
        Rotated Bent Cigar Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = np.array(x) - self._load_shift(2)
        mat = self._load_rot(2)
        f2 = self.bent_cigar_function(np.dot(mat, sum_vectors)) + 200
        return f2

    def cec14_3(self, x: Union[np.array, list]) -> float:
        """
        Rotated Discus Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = np.array(x) - self._load_shift(3)
        mat = self._load_rot(3)
        f3 = self.discuss_function(np.dot(mat, sum_vectors)) + 300
        return f3

    def cec14_4(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Rosenbrock’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = (2.048/100) * (np.array(x) - self._load_shift(4))
        mat = self._load_rot(4)
        f4 = self.rosenbrock_function(np.dot(mat, sum_vectors) + 1) + 400
        return f4

    def cec14_5(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Rosenbrock’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = np.array(x) - self._load_shift(5)
        mat = self._load_rot(5)
        f5 = self.ackley_function(np.dot(mat, sum_vectors)) + 800
        return f5

    def cec14_6(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Weierstrass Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = (0.5/100)*(np.array(x) - self._load_shift(6))
        mat = self._load_rot(6)
        f6 = self.ackley_function(np.dot(mat, sum_vectors)) + 600
        return f6

    def cec14_7(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Griewank’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = (600 / 100) * (np.array(x) - self._load_shift(7))
        mat = self._load_rot(7)
        f7 = self.griewank_function(np.dot(mat, sum_vectors)) + 700
        return f7

    def cec14_8(self, x: Union[np.array, list]) -> float:
        """
        Shifted Rastrigin’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = 5.12 / 100 * (np.array(x) - self._load_shift(8))
        f8 = self.rastrigins_function(sum_vectors) + 800
        return f8

    def cec14_9(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Rastrigin’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = (5.12 / 100) * (np.array(x) - self._load_shift(9))
        mat = self._load_rot(9)

        f9 = self.rastrigins_function(np.dot(mat, sum_vectors)) + 900
        return f9

    def cec14_10(self, x: Union[np.array, list]) -> float:
        """
        Shifted Schwefel’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = 10 * (np.array(x) - self._load_shift(10))
        f10 = self.mod_schwefels_function(sum_vectors) + 1000
        return f10

    def cec14_11(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Schwefel’s Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = 10 * (np.array(x) - self._load_shift(11))
        mat = self._load_rot(11)
        f11 = self.mod_schwefels_function(np.dot(mat, sum_vectors)) + 1100
        return f11

    def cec14_12(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Katsuura Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = 0.05 * (np.array(x) - self._load_shift(12))
        mat = self._load_rot(12)
        f12 = self.katsuura_function(np.dot(mat, sum_vectors)) + 1200
        return f12

    def cec14_13(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated HappyCat Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = 0.05 * (np.array(x) - self._load_shift(13))
        mat = self._load_rot(13)
        f13 = self.happycat_function(np.dot(mat, sum_vectors)) + 1300
        return f13

    def cec14_14(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated HGBat Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = 0.05 * (np.array(x) - self._load_shift(14))
        mat = self._load_rot(14)
        f14 = self.hgbat_function(np.dot(mat, sum_vectors)) + 1400
        return f14

    def cec14_16(self, x: Union[np.array, list]) -> float:
        """
        Shifted and Rotated Expanded Scaffer’s F6 Function
        :param x: solution to be evaluated
        :return: solution evaluation
        :rtype: float
        """

        sum_vectors = np.array(x) - self._load_shift(16)
        mat = self._load_rot(16)
        f16 = self.expanded_scaffer_f6_function(np.dot(mat, sum_vectors) + 1) + 1600
        return f16