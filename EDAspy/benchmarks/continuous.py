#!/usr/bin/env python
# coding: utf-8

import numpy as np


class ContinuousBenchmarkingCEC14:

    def __init__(self, dim):
        assert dim in [2, 10, 20, 30, 50, 100], "Only dimensions 2, 10, 20, 30, 50 and 100 are " \
                                                "available for this benchmarks"
        self.d = dim

    def bent_cigar_function(self, x):
        result = 0
        for i in range(1, self.d):
            result += x[i] ** 2
        result = result * np.power(10, 6) + x[0] ** 2
        return result

    def discuss_function(self, x):
        result = np.power(10, 6) * x[0] ** 2
        for i in range(1, self.d):
            result += x[i] ** 2
        return result

    def rosenbrock_function(self, x):
        result = 0
        for i in range(self.d - 1):
            result += (100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2)
        return result

    def ackley_function(self, x):
        sum1 = sum([x[i] ** 2 for i in range(self.d)])
        sum2 = sum([np.cos(x[i] * 2 * np.pi) for i in range(self.d)])
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / self.d)) - np.exp(sum2 / self.d) + 20 + np.e

    def weierstrass_function(self, x):
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
                aux_f2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)

            f += aux_f1 - self.d * aux_f2

        return f

    def griewank_function(self, x):
        f = 0
        for i in range(self.d):
            f += x[i] ** 2 / 4000
        aux_ = 0
        for i in range(1, self.d):
            aux_ *= np.cos(x[i - 1] / np.sqrt(i)) + 1
        return f - aux_

    def rastrigins_function(self, x):
        result = 0
        for i in range(self.d):
            result += (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10)
        return result

    def high_conditioned_elliptic_function(self, x):
        result = 0
        for i in range(1, self.d + 1):
            result += ((10 ** 6) ** ((i - 1) / (self.d - 1))) * x[i - 1] ** 2
        return result

    def cec14_1(self, x):
        text_file = open("CEC2014/cec14-c-code/input_data/shift_data_1.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = np.array(x) - np.array(shifts[:len(x)]).astype(np.float)

        matrix_file = open("CEC2014/cec14-c-code/input_data/M_1_D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f1 = self.high_conditioned_elliptic_function(np.dot(matrix_file, sum_vectors)) + 100
        return f1

    def cec14_4(self, x):
        text_file = open("input_data/shift_data_4.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = 2.048 / 100 * (np.array(x) - np.array(shifts[:len(x)]).astype(np.float))

        matrix_file = open("input_data/M_4_D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f4 = self.rosenbrock_function(np.dot(matrix_file, sum_vectors) + 1) + 400
        return f4

    def cec14_8(self, x):
        text_file = open("input_data/shift_data_8.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = 5.12 / 100 * (np.array(x) - np.array(shifts[:len(x)]).astype(np.float))

        f8 = self.rastrigins_function(sum_vectors) + 800
        return f8

    def cec14_2(self, x):
        text_file = open("input_data/shift_data_2.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = np.array(x) - np.array(shifts[:len(x)]).astype(np.float)

        matrix_file = open("input_data/M_2_D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f2 = self.bent_cigar_function(np.dot(matrix_file, sum_vectors)) + 200
        return f2

    def cec14_5(self, x):
        text_file = open("input_data/shift_data_5.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = np.array(x) - np.array(shifts[:len(x)]).astype(np.float)

        matrix_file = open("input_data/M_5_D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f5 = self.ackley_function(np.dot(matrix_file, sum_vectors)) + 500
        return f5

    def cec14_6(self, x):
        text_file = open("input_data/shift_data_6.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = (0.5 / 100) * (np.array(x) - np.array(shifts[:len(x)]).astype(np.float))

        matrix_file = open("input_data/M_6_D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f6 = self.ackley_function(np.dot(matrix_file, sum_vectors)) + 600
        return f6

    def cec14_7(self, x):
        text_file = open("input_data/shift_data_7.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = (600 / 100) * (np.array(x) - np.array(shifts[:len(x)]).astype(np.float))

        matrix_file = open("input_data/M_7_D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f7 = self.griewank_function(np.dot(matrix_file, sum_vectors)) + 700
        return f7

    def cec14_9(self, x):
        text_file = open("input_data/shift_data_9.txt", "r")
        shifts = text_file.read().split()
        sum_vectors = (5.12 / 100) * (np.array(x) - np.array(shifts[:len(x)]).astype(np.float))

        matrix_file = open("input_data/M_9D" + str(len(x)) + ".txt", "r")
        matrix_file = matrix_file.read().split()
        matrix_file = np.array(matrix_file).astype(np.float)
        matrix_file = np.reshape(matrix_file, (len(x), len(x)))

        f9 = self.rastrigins_function(np.dot(matrix_file, sum_vectors)) + 900
        return f9
